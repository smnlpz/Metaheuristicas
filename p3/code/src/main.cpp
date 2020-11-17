#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <climits>
#include <random>
#include <stdio.h>
#include <time.h>
#include "./random/random.cpp"

#define SEED 10000

using namespace std;

vector< pair <vector<double>, int> > data; // Dicho pair guardará <atributos[i], clase>
int n_atrib, n_clases;

static const int n_part=5;
vector< pair <vector<double>, int> > particiones[n_part];

vector< pair <vector<double>, int> > train;
vector< pair <vector<double>, int> > test;

//static const int TAM_POB=30;
vector< pair <vector<double>, double> > cromosomas;

default_random_engine generator(SEED); // Para obtener los distintos valores aleatorios necesarios

/*
=============================================================================================
=============================================================================================
					LECTURA DE DATOS Y PREPROCESAMIENTO
=============================================================================================
=============================================================================================
*/

int total_clases(){
	int n=0;
	vector<int> clase;
	for(int i=0; i<data.size(); i++){
		int j=0;
		while(j<clase.size()){
			if(clase[j]==data[i].second) break;
			j++;
		}

		if(j==clase.size()){
			clase.push_back(data[i].second);
			n++;
		}
	}

	return n;
}


void read_data(const char *filename){
	ifstream f(filename);
	string readed;

	if(f){
		do{
			f >> readed;
		}while(readed.compare("@data")!=0); // Devuelve 0 si son iguales

		f >> readed;

		do{ // Mientras no se haya llegado al final del fichero (end of file)
			pair <vector<double>, int> valor;
			int j=0;
			string v;

			while(j<readed.size()){
				if(readed[j]!=','){
					v = v + readed[j];
				}else{
					double atributo=atof(v.c_str());
					valor.first.push_back(atributo);
					v.clear();
				}
				j++;
			}

			valor.second=atoi(v.c_str());

			data.push_back(valor);

			f >> readed;

		}while(!f.eof());

		// ALMACENAMOS LAS VARIABLES GLOBALES
		n_atrib=data[0].first.size();
		n_clases=total_clases();

		f.close();

	}else{
		cout << "Archivo no válido.\n";
		exit(0);
	}
}


void norm_data(){ // Para normalizar los datos

	double maximo, minimo;

	for(int j=0; j<n_atrib; j++){
		vector<double> att;

		for(int i=0; i<data.size(); i++) att.push_back(data[i].first[j]);

		minimo=*min_element(att.begin(), att.end());
		maximo=*max_element(att.begin(), att.end());

		if(maximo!=minimo) // El segundo atributo de los todos datos del fichero ionosphere.arff vale 0
			for(int i=0; i<data.size(); i++)
				data[i].first[j]=(data[i].first[j]-minimo)/(maximo-minimo);
	}

}


void k_folds(){

	shuffle(data.begin(), data.end(), generator);

	for(int i=0; i<n_clases; i++){
		int total_introducidos=0;

		for(int j=0; j<data.size(); j++){
			if(data[j].second == i){
				particiones[total_introducidos%n_part].push_back(data[j]);
				total_introducidos++;
			}
		}
	}
}

void generate_train_test(int particion=-1){
	train.clear();
	test.clear();

	if(particion==-1){
		train.insert(train.end(), data.begin(), data.end());
		test.insert(test.end(), data.begin(), data.end());
	}else{
		for(int i=0; i<n_part; i++){
			if(i!=particion)
				train.insert(train.end(), particiones[i].begin(), particiones[i].end() );
		}

		test.insert(test.end(), particiones[particion].begin(), particiones[particion].end() );
	}
}


/*
=============================================================================================
=============================================================================================
				IMPRESIÓN POR PANTALLA DE DATOS
=============================================================================================
=============================================================================================
*/

void print_data(vector< pair <vector<double>, int> > data_print){
	int cantidad=data_print.size();

	for(int i=0; i<cantidad; i++){
		for(int j=0; j<n_atrib; j++){
			cout << data_print[i].first[j] << " ";
		}
		cout << data_print[i].second << endl;
	}
}


/*
=============================================================================================
=============================================================================================
						1-NN
=============================================================================================
=============================================================================================
*/


double distancia_euclidea(vector<double> a, vector<double> b){
	double dist=0;

	if(a.size()!=b.size()) return -1;

	for(int i=0; i<a.size(); i++){
		dist+=(a[i]-b[i])*(a[i]-b[i]);
	}

	//NO ES NECESARIO CALCULAR LA RAÍZ
	return dist;
}

double distancia(vector<double> a, vector<double> b, vector<double> pesos){
	double dist=0;

	if(a.size()!=b.size()) return -1;

	for(int i=0; i<a.size(); i++){
		if(pesos[i]>=0.2)
			dist+=pesos[i]*(a[i]-b[i])*(a[i]-b[i]);
	}

	//NO ES NECESARIO CALCULAR LA RAÍZ
	return dist;
}

int one_NN(vector<double> e_prima, vector<double> pesos, int indice){ // indice usado para el leave-one-out

	int c_min=train[0].second;
	double d_min=distancia(train[0].first, e_prima, pesos);

	for(int i=1; i<train.size(); i++){
		if(i!=indice){
			double d=distancia(train[i].first, e_prima, pesos);
			if(d<d_min){
				c_min=train[i].second;
				d_min=d;
			}
		}
	}

	return c_min;
}

/*
=============================================================================================
=============================================================================================
					FUNCIÓN DE EVALUACIÓN
=============================================================================================
=============================================================================================
*/

double tasa_clas(vector<double> pesos, bool leave_one_out=false){
	int n=0;
	int indice=-1;

	for(int i=0; i<test.size(); i++){
		if(leave_one_out) indice=i;

		int c_min=one_NN(test[i].first, pesos, indice);

		if(c_min==test[i].second) n++;
	}

	return 100.0*n/test.size();
}

double tasa_red(vector<double> pesos){
	int n=0;

	for(int i=0; i<pesos.size(); i++){
		if(pesos[i]<0.2) n++;
	}

	return 100.0*n/n_atrib;
}

double f_evaluacion(vector<double> pesos, bool print=false, bool leave_one_out=false, double alpha=0.5){
	double clas=tasa_clas(pesos, leave_one_out);
	double red=tasa_red(pesos);

	if(print){
		cout << "Tasa clas: " << clas;
		cout << "\t\tTasa red: " << red << endl;
		//cout << endl << clas << "\t" << red << endl;
	}

	return alpha*clas+(1-alpha)*red;
}


/*
=============================================================================================
=============================================================================================
						BÚSQUEDA LOCAL
=============================================================================================
=============================================================================================
*/

vector<double> generar_solucion_inicial(){
	uniform_real_distribution<double> distribution(0.0,1.0);

	vector<double> pesos;

	for(int i=0; i<n_atrib; i++)	pesos.push_back(distribution(generator));

	return pesos;
}

vector<int> generar_indices(int cuantos){
	vector<int> indices;

	for(int i=0; i<cuantos; i++)	indices.push_back(i);

	return indices;
}

void mutacion(vector<double> &pesos, int i){
	normal_distribution<double> distribution(0.0,0.3);

	pesos[i]+=distribution(generator);

	if(pesos[i]>1) pesos[i]=1;
	if(pesos[i]<0) pesos[i]=0;

}


vector<double> busqueda_local(double alpha, int max_eval, int &evals, vector<double> pesos){
	int n_generados=0, n_eval=0;
	int max_generados=20*n_atrib;
	//int max_eval=15000;

	if(pesos.empty())	pesos=generar_solucion_inicial();

	vector<int> indices=generar_indices(n_atrib);
	int i=0;

	double valor_eval=0;

	while(n_generados < max_generados && n_eval < max_eval){
		if(i==0)	shuffle(indices.begin(), indices.end(), generator);

		vector<double> tmp=pesos;

		mutacion(tmp, indices[i]);
		n_generados++;

		double new_valor=f_evaluacion(tmp, false, true, alpha);
		n_eval++;

		if(new_valor>valor_eval){
			pesos=tmp;
			valor_eval=new_valor;
			n_generados=0;
		}

		i++;
		if(i==n_atrib)	i=0;

	}

	evals+=n_eval;

	return pesos;
}

/*
=============================================================================================
=============================================================================================
						PRÁCTICA 2
=============================================================================================
=============================================================================================
*/

bool comp_cromosoma(const pair< vector<double>, double> &c1, const pair< vector<double>, double> &c2){
	return c1.second > c2.second;
}

void reordenar(){
	sort(cromosomas.begin(), cromosomas.end(), comp_cromosoma);
}

void generate_cromosomas(int tam){
	cromosomas.clear();
	pair< vector<double>, double> tmp;

	for(int i=0; i<tam; i++){
		tmp.first=generar_solucion_inicial();
		tmp.second=f_evaluacion(tmp.first, false, true);
		cromosomas.push_back(tmp);
		tmp.first.clear();
	}

	//reordenar();
}

void print_crom(){
	for(int i=0; i<cromosomas.size(); i++){
		//for(int j=0; j<n_atrib; j++){
		//	cout << cromosomas[i].first[j] << " ,";
		//}
		cout << cromosomas[i].second << ", ";
	}
	cout << endl;
}

/*
=============================================================================================
=============================================================================================
						PRÁCTICA 3
=============================================================================================
=============================================================================================
*/

/*
=============================================================================================
=============================================================================================
						ENFRIAMIENTO SIMULADO
=============================================================================================
=============================================================================================
*/


/*void enfriamiento(double &T, double T0, double TF, double M){
	double beta=(T0-TF)/(M*T0*TF);
	T=T/(1+beta*T);
}*/

vector<double> ES(){
	vector<double> solucion_actual=generar_solucion_inicial();
	double valor_actual=f_evaluacion(solucion_actual, false, true);
	vector<double> mejor_solucion=solucion_actual;
	double mejor_valor=valor_actual;

	//double phi=0.3;
	double mu=0.3;
	double minus_ln_phi=1.20397280433;
	double T0=mu*valor_actual/minus_ln_phi;
	double TF=0.001;
	double Tactual=T0;
	int K=1;
	if(TF>T0)	return mejor_solucion;

	// n_atrib es una variable global
	int max_vecinos=10*n_atrib, max_exitos=0.1*max_vecinos;
	int n_eval=0, max_eval=15000;
	int M=max_eval/max_vecinos;	// Enfriamientos

	double beta=(T0-TF)/(M*T0*TF);

	uniform_real_distribution<double> distribution(0.0,1.0);

	int n_vecinos, n_exitos;
	do{
		n_vecinos=0, n_exitos=0;

		while(n_vecinos<max_vecinos && n_exitos<max_exitos && n_eval<max_eval){
			vector<double> tmp=solucion_actual;

			int i=Randint(0,n_atrib-1);
			mutacion(tmp, i);
			n_vecinos++;

			double new_valor=f_evaluacion(tmp, false, true);
			n_eval++;

			double increment=new_valor-valor_actual;

			if(increment==0)	increment=0.005;

			double value=exp(increment/(Tactual*K));

			// Comprobamos con increment > 0 porque estamos en un problema de maximización
			// Si increment=0, la exponencial nos dara 1 y entrará dentro del if
			if(increment>0 || (distribution(generator) <= value)){
				solucion_actual=tmp;
				valor_actual=new_valor;
				n_exitos++;
				if(valor_actual>mejor_valor){
					mejor_solucion=solucion_actual;
					mejor_valor=valor_actual;
				}
			}
		}

		//Tactual=Tactual/(1+beta*Tactual);
		Tactual*=0.9;

		//cout << Tactual << " " << n_eval << " " << mejor_valor << " " << n_exitos << " " << " " << n_vecinos << " " << K << endl << endl;
	}while(Tactual>TF && n_exitos>0 && n_eval<max_eval);

	return mejor_solucion;
}

/*
=============================================================================================
=============================================================================================
						BÚSQUEDA LOCAL REITERADA
=============================================================================================
=============================================================================================
*/

vector<double> ILS(){
	vector<double> solucion_actual=generar_solucion_inicial();
	int unused=0;
	vector<double> mejor_solucion=busqueda_local(0.5, 1000, unused, solucion_actual);
	double mejor_valor=f_evaluacion(mejor_solucion, false, true);

	vector<int> indices=generar_indices(n_atrib);
	int t=0.1*n_atrib;

	for(int i=1; i<15; i++){
		shuffle(indices.begin(), indices.end(), generator);
		vector<double> tmp=mejor_solucion;

		for(int i=0; i<t; i++){
			mutacion(tmp, indices[i]);
		}

		tmp=busqueda_local(0.5, 1000, unused, tmp);

		double new_valor=f_evaluacion(tmp, false, true);

		if(new_valor>mejor_valor){
			mejor_valor=new_valor;
			mejor_solucion=tmp;
		}
	}

	return mejor_solucion;
}

/*
=============================================================================================
=============================================================================================
						EVOLUCIÓN DIFERENCIAL
=============================================================================================
=============================================================================================
*/

void mutacion_rand(vector<double> &pesos, int p1, int p2, int p3, int i){
	double F=0.5;

	double valor=cromosomas[p1].first[i] + F * (cromosomas[p2].first[i] - cromosomas[p3].first[i]);

	pesos[i]=cromosomas[p1].first[i] + F * (cromosomas[p2].first[i] - cromosomas[p3].first[i]);

	if(pesos[i]>1) pesos[i]=1;
	if(pesos[i]<0) pesos[i]=0;
}

vector<double> DE_rand_1(){
	int generacion=0;
	int tam_pob=50;
	double prob_cruce=0.5;
	int eval=0, max_eval=15000;

	generate_cromosomas(tam_pob);
	eval+=tam_pob;

	//reordenar();

	while(eval<15000){
		vector< pair<vector<double>,double> > hijos;
		for(int i=0; i<tam_pob; i++){
			vector<int> indices=generar_indices(n_atrib);
			shuffle(indices.begin(), indices.end(), generator);

			int p1, p2, p3;
			do{
				p1=Randint(0,tam_pob-1);
			}while(p1==i);
			do{
				p2=Randint(0,tam_pob-1);
			}while(p2==i || p2==p1);
			do{
				p3=Randint(0,tam_pob-1);
			}while(p3==i || p3==p2 || p3==p1);

			pair<vector<double>,double> hijo_generado;
			hijo_generado.first=cromosomas[i].first;

			for(int j=0; j<(int)(prob_cruce*n_atrib); j++){
				mutacion_rand(hijo_generado.first, p1, p2, p3, indices[j]);
			}


			hijo_generado.second=f_evaluacion(hijo_generado.first, false, true);
			//cout << eval << endl;
			eval++;
			hijos.push_back(hijo_generado);
			//cout << hijo_generado.second << endl;
		}

		for(int i=0; i<tam_pob; i++){
			if(hijos[i].second>cromosomas[i].second)
				cromosomas[i]=hijos[i];
		}

	}

	reordenar();

	return cromosomas[0].first;
}


/*************************************************************************/

void mutacion_current_to_best(vector<double> &pesos, int p1, int p2, int current, int i){
	double F=0.5;

	//double valor=cromosomas[current].first[i]+F*(cromosomas[0].first[i]-cromosomas[current].first[i])+F*(cromosomas[p1].first[i]-cromosomas[p2].first[i]);
	pesos[i]=cromosomas[current].first[i]+F*(cromosomas[0].first[i]-cromosomas[current].first[i])+F*(cromosomas[p1].first[i]-cromosomas[p2].first[i]);
	//cout << current << endl;
	if(pesos[i]>1) pesos[i]=1;
	if(pesos[i]<0) pesos[i]=0;
}

vector<double> DE_current_to_best(){
	int generacion=0;
	int tam_pob=50;
	double prob_cruce=0.5;
	int eval=0, max_eval=15000;

	generate_cromosomas(tam_pob);
	eval+=tam_pob;
	reordenar();

	while(eval<15000){
		vector< pair<vector<double>,double> > hijos;
		for(int i=0; i<tam_pob; i++){
			vector<int> indices=generar_indices(n_atrib);
			shuffle(indices.begin(), indices.end(), generator);

			int p1, p2;
			do{
				p1=Randint(0,tam_pob-1);
			}while(p1==i);
			do{
				p2=Randint(0,tam_pob-1);
			}while(p2==i || p2==p1);

			pair<vector<double>,double> hijo_generado;
			hijo_generado.first=cromosomas[i].first;

			for(int j=0; j<(int)(prob_cruce*n_atrib); j++){
				mutacion_current_to_best(hijo_generado.first, p1, p2, i, indices[j]);
			}

			hijo_generado.second=f_evaluacion(hijo_generado.first, false, true);
			eval++;

			hijos.push_back(hijo_generado);
			//cout << hijo_generado.second << endl;
		}

		for(int i=0; i<tam_pob; i++){
			if(hijos[i].second>cromosomas[i].second)
				cromosomas[i]=hijos[i];
		}
		reordenar();
		//cout << endl;
		//print_crom();
	}

	return cromosomas[0].first;
}



/*
=============================================================================================
=============================================================================================
					FUNCIONES PARA EL TIEMPO
=============================================================================================
=============================================================================================
*/


clock_t start_time;

double elapsed;

void start_timers()
/*
      FUNCTION:       virtual and real time of day are computed and stored to
                      allow at later time the computation of the elapsed time
		      (virtual or real)
      INPUT:          none
      OUTPUT:         none
      (SIDE)EFFECTS:  virtual and real time are computed
*/
{ start_time = clock(); }

double elapsed_time()
//	TIMER_TYPE type;
/*
      FUNCTION:       return the time used in seconds (virtual or real, depending on type)
      INPUT:          TIMER_TYPE (virtual or real time)
      OUTPUT:         seconds since last call to start_timers (virtual or real)
      (SIDE)EFFECTS:  none
*/
{ elapsed = clock()- start_time; return elapsed / CLOCKS_PER_SEC; }




/*
=============================================================================================
=============================================================================================
						MAIN
=============================================================================================
=============================================================================================
*/

int main(int argc, char *argv[]){
	Set_random(SEED);

	read_data(argv[1]);
	norm_data();
	k_folds();


	cout << "----------------------------- Datos usados: " << argv[1] << " -------------------------------" << endl << endl;

	for(int i=0; i<n_part; i++){
		generate_train_test(i);

		cout << "=============================================================================================" << endl;
		cout << "=============================================================================================" << endl;
		cout << "                                     PARTICION " << i+1 << endl;
		cout << "=============================================================================================" << endl;
		cout << "=============================================================================================" << endl << endl;


		cout << "ES:\t\t";

		start_timers();

		vector<double> pesos_es=ES();

		double valor_es=f_evaluacion(pesos_es, true, false);

		double tiempo_es=elapsed_time();

		cout << "*********************** F evaluacion: " << valor_es << " (tiempo: " << tiempo_es << ") ****************************" << endl << endl;


		cout << "ILS:\t\t";

		start_timers();

		vector<double> pesos_ils=ILS();

		double valor_ils=f_evaluacion(pesos_ils, true, false);

		double tiempo_ils=elapsed_time();

		cout << "*********************** F evaluacion: " << valor_ils << " (tiempo: " << tiempo_ils << ") ****************************" << endl << endl;


		cout << "DE_RAND_1:\t\t";

		start_timers();

		vector<double> pesos_de_rand=DE_rand_1();

		double valor_de_rand=f_evaluacion(pesos_de_rand, true, false);

		double tiempo_de_rand=elapsed_time();

		cout << "*********************** F evaluacion: " << valor_de_rand << " (tiempo: " << tiempo_de_rand << ") ****************************" << endl << endl;


		cout << "DE_CURRENT_TO_BEST:\t\t";

		start_timers();

		vector<double> pesos_de_ctb=DE_current_to_best();

		double valor_de_ctb=f_evaluacion(pesos_de_ctb, true, false);

		double tiempo_de_ctb=elapsed_time();

		cout << "*********************** F evaluacion: " << valor_de_ctb << " (tiempo: " << tiempo_de_ctb << ") ****************************" << endl << endl;

	}


	cout << "                                          FIN                                          " << endl;
}
