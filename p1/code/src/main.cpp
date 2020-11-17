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
		cout << "                 Tasa red: " << red << endl;
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

vector<int> generar_indices(){
	vector<int> indices;
	
	for(int i=0; i<n_atrib; i++)	indices.push_back(i);
	
	return indices;
}

void mutacion(vector<double> &pesos, int i){
	normal_distribution<double> distribution(0.0,0.3);
	
	pesos[i]+=distribution(generator);
	
	if(pesos[i]>1) pesos[i]=1;
	if(pesos[i]<0) pesos[i]=0;

}


vector<double> busqueda_local(double alpha=0.5){	
	int n_generados=0, n_eval=0;
	int max_generados=20*n_atrib, max_eval=15000;
	
	vector<double> pesos=generar_solucion_inicial();
	vector<int> indices=generar_indices();
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
	
	return pesos;
}


/*
=============================================================================================
=============================================================================================
						GREEDY RELIEF
=============================================================================================
=============================================================================================
*/

int close_enemy_index(pair< vector<double>, int> e){
	int ind_min=-1;
	double d_min=numeric_limits<double>::max();
	
	for(int i=0; i<train.size(); i++){
		if(train[i].second!=e.second){
			double d_tmp=distancia_euclidea(e.first, train[i].first);
			
			if(d_tmp<d_min){
				ind_min=i;
				d_min=d_tmp;
			}
		}
	}
	
	return ind_min;
}

int close_friend_index(pair< vector<double>, int> e, int indice){
	int ind_min=-1;
	double d_min=numeric_limits<double>::max();
	
	for(int i=0; i<train.size(); i++){
		if(train[i].second==e.second && i!=indice){
			double d_tmp=distancia_euclidea(e.first, train[i].first);
			
			if(d_tmp<d_min){
				ind_min=i;
				d_min=d_tmp;
			}
		}
	}
	
	return ind_min;
}

vector<double> relief(){
	vector<double> pesos(n_atrib, 0.0); // Inicializamos todos los valores a 0
	
	for(int i=0; i<train.size(); i++){
		int close_enemy_ind=close_enemy_index(train[i]);
		int close_friend_ind=close_friend_index(train[i], i);
		
		for(int j=0; j<n_atrib; j++){
			pesos[j] = pesos[j] + abs(train[i].first[j]-train[close_enemy_ind].first[j]) 								- abs(train[i].first[j]-train[close_friend_ind].first[j]);	
		}
		
	}
	
	double peso_max=*max_element(pesos.begin(), pesos.end());

	for(int j=0; j<n_atrib; j++){
		if(pesos[j]<0)
			pesos[j]=0;
		if(pesos[j]>1)
			pesos[j]=pesos[j]/peso_max;
	}	
		
	return pesos;
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
{
    start_time = clock();
}



double elapsed_time()
//	TIMER_TYPE type;
/*    
      FUNCTION:       return the time used in seconds (virtual or real, depending on type) 
      INPUT:          TIMER_TYPE (virtual or real time)
      OUTPUT:         seconds since last call to start_timers (virtual or real)
      (SIDE)EFFECTS:  none
*/
{
    elapsed = clock()- start_time;
    return elapsed / CLOCKS_PER_SEC;
}



/*
=============================================================================================
=============================================================================================
						MAIN
=============================================================================================
=============================================================================================
*/

int main(int argc, char *argv[]){
	
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
		
		cout << "KNN:         ";
		
		start_timers();
		
		vector<double> pesos_knn(n_atrib, 1.0); // Inicializamos todos los valores a 1

		double valor_knn=f_evaluacion(pesos_knn, true, false); // Valor de f sin ajustar los pesos
		
		double tiempo_knn=elapsed_time();
				
		cout << "*********************** F evaluacion: " << valor_knn << " (tiempo: " << tiempo_knn << ") ****************************" << endl << endl;
	
			
		cout << "BL:          ";
		
		start_timers();
		
		vector<double> pesos_bl=busqueda_local();
	
		double valor_bl=f_evaluacion(pesos_bl, true, false); // Valor de f sin ajustar los pesos
		
		double tiempo_bl=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_bl << " (tiempo: " << tiempo_bl << ") ****************************" << endl << endl;
		
		
		cout << "RELIEF:      ";
		
		start_timers();
		
		vector<double> pesos_relief=relief();

		double valor_relief=f_evaluacion(pesos_relief, true, false);
	
		double tiempo_relief=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_relief << " (tiempo: " << tiempo_relief << ") ****************************" << endl << endl;
		
			
		cout << "BL (alpha=1):    ";
		
		start_timers();
		
		vector<double> pesos_bl_1=busqueda_local(1.0);
	
		double valor_bl_1=f_evaluacion(pesos_bl_1, true, false); // Valor de f sin ajustar los pesos
		
		double tiempo_bl_1=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_bl_1 << " (tiempo: " << tiempo_bl_1 << ") ****************************" << endl << endl;
		
		
		cout << "BL (alpha=0):    ";
		
		start_timers();
		
		vector<double> pesos_bl_0=busqueda_local(0.0);
	
		double valor_bl_0=f_evaluacion(pesos_bl_0, true, false); // Valor de f sin ajustar los pesos
		
		double tiempo_bl_0=elapsed_time();
		
		cout << "*********************** F evaluacion: " << valor_bl_0 << " (tiempo: " << tiempo_bl_0 << ") ****************************" << endl << endl;
		
		cout << endl << endl;
		
	}
	
	
	cout << "                                          FIN                                          " << endl;
	
}
