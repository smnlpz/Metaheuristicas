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

/*
static const double P_CRUCE=0.7;

static const double P_MUTACION=0.001;
*/

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
	
	/*vector<int> indices;
	
	for(int i=0; i<pesos.size(); i++){
		if(pesos[i]>=0.2){
			indices.push_back(i);
		}
	}*/
	
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
		for(int j=0; j<n_atrib; j++){
			cout << cromosomas[i].first[j] << " ,";
		}
		cout << cromosomas[i].second << ", \t";
	}
}

vector< pair< vector<double>, double > > BLX_alpha(int c1, int c2, double alpha=0.3){ //c1 y c2 índices de los padres escogidos
	
	vector< pair< vector<double>, double > > hijos;
	
	hijos.resize(2);
	
	for(int i=0; i<n_atrib; i++){
		double c_max=max(cromosomas[c1].first[i], cromosomas[c2].first[i]);
		double c_min=min(cromosomas[c1].first[i], cromosomas[c2].first[i]);
		double I=c_max-c_min;
		
		double add1=Randfloat( c_min - I*alpha, c_max + I*alpha );
		double add2=Randfloat( c_min - I*alpha, c_max + I*alpha );
		
		if(add1>1.0) hijos[0].first.push_back(1.0);
		else if(add1<0.0) hijos[0].first.push_back(0.0);
		else hijos[0].first.push_back(add1);
		
		if(add2>1.0) hijos[1].first.push_back(1.0);
		else if(add2<0.0) hijos[1].first.push_back(0.0);
		else hijos[1].first.push_back(add2);
	}
	
	hijos[0].second=f_evaluacion(hijos[0].first, false, true);
	hijos[1].second=f_evaluacion(hijos[1].first, false, true);
	
	return hijos;
}

vector< pair<vector<double>, double> > arithmetic_crossover(int c1, int c2){ //c1 y c2 índices de los padres escogidos
	vector< pair< vector<double>, double > > hijos;
	
	hijos.resize(2);
	
	double alpha=0;
	
	for(int i=0; i<n_atrib; i++){
		alpha=Randfloat(0,1);
		hijos[0].first.push_back( alpha*cromosomas[c1].first[i] + (1.0-alpha)*cromosomas[c2].first[i] );
		hijos[1].first.push_back( (1.0-alpha)*cromosomas[c1].first[i] + alpha*cromosomas[c2].first[i] );
	}
	
	hijos[0].second=f_evaluacion(hijos[0].first, false, true);
	hijos[1].second=f_evaluacion(hijos[1].first, false, true);
	
	return hijos;
}

int binary_tournament(vector<int> &indices){ // Usamos un vector de indices para no repetir los padres
	int competidor1=Randint(0, indices.size()-1);
	int competidor2;
	
	do{
		competidor2=Randint(0, indices.size()-1);
	}while(competidor1==competidor2);
	
	if(cromosomas[indices[competidor1]].second > cromosomas[indices[competidor2]].second){
		indices.erase(indices.begin()+competidor1);
		return competidor1;
	}else{
		indices.erase(indices.begin()+competidor2);
		return competidor2;
	}
}

int binary_tournament(){
	int competidor1=Randint(0, cromosomas.size()-1);
	int competidor2;
	
	do{
		competidor2=Randint(0, cromosomas.size()-1);
	}while(competidor1==competidor2);
	
	if(cromosomas[competidor1].second > cromosomas[competidor2].second)	return competidor1;
	else		return competidor2;
}

/*
=============================================================================================
=============================================================================================
						GENÉTICOS
=============================================================================================
=============================================================================================
*/

vector<double> AGG(int tipo_cruce){ // cruce=0 -----> BLXalpha; cruce=1 -----> arithmetic crossover
	int generacion=0;
	double prob_cruce=0.7;
	double prob_mut=0.001;
	int tam_pob=30;
	int n_parejas=tam_pob/2;
	int evals=0;
	
	generate_cromosomas(tam_pob);
	evals+=tam_pob;
	
	reordenar();

	while(evals<15000){
		vector< pair<vector<double>,double> > hijos;
		vector<int> indices=generar_indices(tam_pob);
		
		// Generamos la nueva poblacion
		for(int i=0; i<(int)(n_parejas*prob_cruce); i++){  
			int padre1=binary_tournament(indices);
			int padre2=binary_tournament(indices);
			
			if(tipo_cruce==0){
				vector< pair< vector<double>, double > > tmp=BLX_alpha(padre1,padre2);
				hijos.push_back(tmp[0]);
				hijos.push_back(tmp[1]);
			}else if(tipo_cruce==1){
				vector< pair< vector<double>, double > > tmp=arithmetic_crossover(padre1,padre2);
				hijos.push_back(tmp[0]);
				hijos.push_back(tmp[1]);
			}
			
			evals+=2;
		}
		
		// MUTACION
		for(int i=0; i<(int)hijos.size()*n_atrib*prob_mut; i++){
			//double Mu_next += ceil (log(Rand()) / log(1.0 - prob_mut));
			//Se determina el cromosoma y el gen que corresponden a la posicion que se va a mutar
			//int crom_mut=Mu_next/n_atrib;
			//int gen_mut=Mu_next%n_atrib;
			//int gen_mut=fmod(Mu_next,n_atrib);
			int crom_mut=Randint(0, hijos.size()-1);
			int gen_mut=Randint(0, n_atrib-1);
			
			mutacion(hijos[crom_mut].first, gen_mut);
			hijos[crom_mut].second=f_evaluacion(hijos[crom_mut].first, false, true);
			evals++;
		}
		
		// Añadimos los que no han sido padres
		for(int i=0; i<indices.size(); i++)	hijos.push_back(cromosomas[indices[i]]);
		
		double minimo=hijos[0].second;
		int ind_min=0;
		
		for(int i=1; i<hijos.size(); i++){
			if(hijos[i].second < minimo){
				minimo=hijos[i].second;
				ind_min=i;
			}
		}
		
		hijos[ind_min]=cromosomas[0];
		
		cromosomas=hijos;
						
		generacion++;
		
		reordenar();
	}
	
	return cromosomas[0].first;
}

vector<double> AGE(int tipo_cruce){
	int generacion=0;
	double pm_gen=0.001;
	double pm_cromosoma=pm_gen*n_atrib;
	int tam_pob=30;
	int evals=0;
	
	generate_cromosomas(tam_pob);
	evals+=tam_pob;
	
	reordenar();

	while(evals<15000){
		int padre1=binary_tournament();
		int padre2=binary_tournament();
		vector< pair<vector<double>,double> > hijos;
		
		if(tipo_cruce==0){
			hijos=BLX_alpha(padre1,padre2);
		}else if(tipo_cruce==1){
			hijos=arithmetic_crossover(padre1,padre2);
		}
		
		evals+=2;
		
		if(Rand() < pm_cromosoma){
			int gen_mut=Randint(0, n_atrib-1);
			mutacion(hijos[0].first, gen_mut);
			hijos[0].second=f_evaluacion(hijos[0].first, false, true);
			evals++;
		}
		
		if(Rand() < pm_cromosoma){
			int gen_mut=Randint(0, n_atrib-1);
			mutacion(hijos[1].first, gen_mut);
			hijos[1].second=f_evaluacion(hijos[1].first, false, true);
			evals++;
		}
		
		
		cromosomas.push_back(hijos[0]);
		cromosomas.push_back(hijos[1]);
		
		reordenar();
		
		cromosomas.resize(tam_pob);
		
		generacion++;
		
		/*
		if(hijos[0].second > hijos[1].second){
			if(hijos[0].second >= cromosomas[tam_pob-2].second){
				cromosomas[tam_pob-2]=hijos[0];
				if(hijos[1].second >= cromosomas[tam_pob-1].second){
					cromosomas[tam_pob-1]=hijos[1];
				}
			}else if(hijos[0].second >= cromosomas[tam_pob-1].second){
				cromosomas[tam_pob-1]=hijos[0];
			}
		}else{
			if(hijos[1].second >= cromosomas[tam_pob-2].second){
				cromosomas[tam_pob-2]=hijos[1];
				if(hijos[0].second >= cromosomas[tam_pob-1].second){
					cromosomas[tam_pob-1]=hijos[0];
				}
			}else if(hijos[1].second >= cromosomas[tam_pob-1].second){
				cromosomas[tam_pob-1]=hijos[1];
			}
		}*/
		
	}
	
	return cromosomas[0].first;
}


/*
=============================================================================================
=============================================================================================
						MEMÉTICOS
=============================================================================================
=============================================================================================
*/

vector<double> AM(int n_gen_bl, double porcentaje_pob, bool mejor){ // cruce = Arithmetic-crossover;
	int generacion=0;
	double prob_cruce=0.7;
	double prob_mut=0.001;
	int tam_pob=10;
	int n_parejas=tam_pob/2;
	int evals=0;
	
	generate_cromosomas(tam_pob);
	evals+=tam_pob;
	
	reordenar();

	while(evals<15000){
		if(generacion % n_gen_bl==0){
			if(mejor){
				for(int i=0; i<(int)tam_pob*porcentaje_pob; i++){
					cromosomas[i].first = busqueda_local(0.5, 2*n_atrib, evals, cromosomas[i].first);
					cromosomas[i].second = f_evaluacion(cromosomas[i].first, false, true);
					evals++;
				}
			}else{
				vector<int> usados;
				for(int i=0; i<(int)tam_pob*porcentaje_pob; i++){
					int cual;
					do{
						cual=Randint(0, tam_pob-1);
					}while(find(usados.begin(), usados.end(), cual) != usados.end());
					
					usados.push_back(cual);
					
					cromosomas[cual].first = busqueda_local(0.5, 2*n_atrib, evals, cromosomas[cual].first);
					cromosomas[cual].second = f_evaluacion(cromosomas[cual].first, false, true);
					evals++;
				}
				//for(int i=0; i<usados.size(); i++) cout << usados[i] << "\t";
				//cout << endl;
			}
			reordenar();
		}

		vector< pair<vector<double>,double> > hijos;
		vector<int> indices=generar_indices(tam_pob);
		
		// Generamos la nueva poblacion
		for(int i=0; i<(int)(n_parejas*prob_cruce); i++){  
			int padre1=binary_tournament(indices);
			int padre2=binary_tournament(indices);
			
			vector< pair< vector<double>, double > > tmp=arithmetic_crossover(padre1,padre2);
			hijos.push_back(tmp[0]);
			hijos.push_back(tmp[1]);
						
			evals+=2;
		}
		
		// MUTACION
		for(int i=0; i<(int)hijos.size()*n_atrib*prob_mut; i++){
			int crom_mut=Randint(0, hijos.size()-1);
			int gen_mut=Randint(0, n_atrib-1);
			
			mutacion(hijos[crom_mut].first, gen_mut);
			hijos[crom_mut].second=f_evaluacion(hijos[crom_mut].first, false, true);
			evals++;
		}
		
		// Añadimos los que no han sido padres
		for(int i=0; i<indices.size(); i++)	hijos.push_back(cromosomas[indices[i]]);
		
		double minimo=hijos[0].second;
		int ind_min=0;
		
		for(int i=1; i<hijos.size(); i++){
			if(hijos[i].second < minimo){
				minimo=hijos[i].second;
				ind_min=i;
			}
		}
		
		hijos[ind_min]=cromosomas[0];
		
		cromosomas=hijos;
						
		generacion++;
		
		reordenar();
		
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
		
		
		cout << "AGG BLX:\t\t";
		
		start_timers();
		
		vector<double> pesos_agg_blx=AGG(0);
		
		double valor_agg_blx=f_evaluacion(pesos_agg_blx, true, false);
		
		double tiempo_agg_blx=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_agg_blx << " (tiempo: " << tiempo_agg_blx << ") ****************************" << endl << endl;
		
		
		
		
		
		
			
		cout << "AGG ARITHMETIC:\t\t";
		
		start_timers();
		
		vector<double> pesos_agg_arith=AGG(1);
	
		double valor_agg_arith=f_evaluacion(pesos_agg_arith, true, false); 
		
		double tiempo_agg_arith=elapsed_time();
				
		cout << "*********************** F evaluacion: " << valor_agg_arith << " (tiempo: " << tiempo_agg_arith << ") ****************************" << endl << endl;
		
		
		
		
				
		cout << "AGE BLX:\t\t";
		
		start_timers();
		
		vector<double> pesos_age_blx=AGE(0);
	
		double valor_age_blx=f_evaluacion(pesos_age_blx, true, false); 
		
		double tiempo_age_blx=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_age_blx << " (tiempo: " << tiempo_age_blx << ") ****************************" << endl << endl;
		
		
		
		
		
		
			
		cout << "AGE ARITHMETIC:\t\t";
		
		start_timers();
		
		vector<double> pesos_age_arith=AGE(1);
	
		double valor_age_arith=f_evaluacion(pesos_age_arith, true, false); 
		
		double tiempo_age_arith=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_age_arith << " (tiempo: " << tiempo_age_arith << ") ****************************" << endl << endl;
		
		
		
		
		
		
			
		cout << "AM(10, 1.0):\t\t";
		
		start_timers();
		
		vector<double> pesos_am_1=AM(10, 1.0, true);
		
		double valor_am_1=f_evaluacion(pesos_am_1, true, false);
		
		double tiempo_am_1=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_am_1 << " (tiempo: " << tiempo_am_1 << ") ****************************" << endl << endl;
		
		
		
		
		
		
			
		cout << "AM(10, 0.1):\t\t";
		
		start_timers();
		
		vector<double> pesos_am_2=AM(10, 0.1, false);
		
		double valor_am_2=f_evaluacion(pesos_am_2, true, false);
		
		double tiempo_am_2=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_am_2 << " (tiempo: " << tiempo_am_2 << ") ****************************" << endl << endl;
		
		
		
		
		
		
			
		cout << "AM(10, 0.1mej):\t\t";
		
		start_timers();
		
		vector<double> pesos_am_3=AM(10, 0.1, true);
		
		double valor_am_3=f_evaluacion(pesos_am_3, true, false);
		
		double tiempo_am_3=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_am_3 << " (tiempo: " << tiempo_am_3 << ") ****************************" << endl << endl;
		
		
		
		
		
		
		
		
		cout << "AM(1, 0.1mej):\t\t";
		
		start_timers();
		
		vector<double> pesos_am_4=AM(1, 0.1, true);
		
		double valor_am_4=f_evaluacion(pesos_am_4, true, false);
		
		double tiempo_am_4=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_am_4 << " (tiempo: " << tiempo_am_4 << ") ****************************" << endl << endl;
		
	}	
	cout << "                                          FIN                                          " << endl;
	
}


