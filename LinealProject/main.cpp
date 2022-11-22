/*
 * FEcha: 21/09/2022
 * AUtor: Carlos Rojas
 * Materia: HPC
 * Topico: IMplementacion de la regresion lineal como
 * modelo de c++
 * REquerimientos_
 * - Construir ina clase EXtraction, que permita
 * manipular, extraer y cargar los datos.
 * - Construir una clase Lineal Regresion, que permita
 * los calculos de la funcion de costo, gradiente decendiente
 * entre otras.*/

#include "ClassExtraction/extractiondata.h"
#include "Regresion/linearregression.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>


int main(int argc, char* argv[])
{

    /*Se crea un objeto del tipo ClassExtracition*/
   ExtractionData ExData(argv[1],argv[2],argv[3]);

       //SE instancia la clase  de regresion lineal en un objeto
   LinearRegression modeloLR;


   /*Se crea un vectr de vectores del tipo String para cargar  el objecto ExData lectura*/
   std::vector<std::vector<std::string>> dataframe = ExData.LeerCSV();

   /* cantidad de filas y columnas*/
    int filas = dataframe.size();
    int columnas = dataframe[0].size();


    /*Se crea una matrix Eigen para ingresar los valores a esa matriz*/
    Eigen::MatrixXd matData = ExData.CSVtoEigen(dataframe, filas, columnas);

    /*SE normaliza la matriz de datos*/

    Eigen::MatrixXd matNorm = ExData.Norm(matData);

    /*SE divide en datos de entrenaniento y datos de prueba*/
    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> div_datos = ExData.TrainTestSplit(matNorm, 0.8);
    /*Se descomprime la dupla en 4 conjuntos*/

    std::tie(X_train, y_train, X_test, y_test) = div_datos;
    //SE creA VECTORES AUXILIARES PARA PRUEBA Y ENTRENAMIENTO INICIAÑLIZADO EN 1
    Eigen::VectorXd V_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd V_test = Eigen::VectorXd::Ones(X_test.rows());

    // Se redimensiona la matriz de entrenamiento y prueba  para ser ajustadas a los vectores au
    // auxiliares anteriores

    X_train.conservativeResize(X_train.rows(),X_train.cols()+1);
    X_train.col(X_train.cols()-1) = V_train;
    X_test.col(X_test.cols()-1) = V_test;

    //Se crea el vector de coefisientes theta
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    Eigen::VectorXd thetaTest = Eigen::VectorXd::Zero(X_test.cols());
    // Se establece el alpha como ratio de aprendizaje  de tipo flotante
    float alpha = 0.01;
    int num_iter = 1000;

    /* Se cre un vector para almacenar las tetas de salida(parametros m y b)*/
    Eigen::VectorXd thetasOut;
    Eigen::VectorXd thetasOutTest;
    // SE cre un vector sensillo (std) de flotantes para almacenar los valores de costo
    std::vector<float> costo;
    std::vector<float> costoTest;
    //Se calcula el gradiente descendiente
    std::tuple<Eigen::VectorXd, std::vector<float>> g_descendiente = modeloLR.GradientDescent(X_train,y_train,theta, alpha, num_iter);
    std::tuple<Eigen::VectorXd, std::vector<float>> g_descendienteTest = modeloLR.GradientDescent(X_test,y_test,thetaTest, alpha, num_iter);
    //Se desempaqueta el gradiente
    std::tie(thetasOut,costo) = g_descendiente;
    std::tie(thetasOutTest,costoTest) = g_descendienteTest;
    //Se almacenan los valores de thetas y costos en un fichero apra ser visualizado
    //ExData.VectortoFile(costo, "costos.txt");
    //ExData.EigentoFile(thetasOut,"thetas.txt");
    //----------------------------------------------------------------
    // S extrae el promedio de la matriz de entrada

    auto prom_data = ExData.Promedio(matData);
    //SE extraen los valores de las variables independientes
    //ACA PQ ESTE DATA SET DE 0 A 11 SON INDEPENDIETES, PERO ESTE 11 CAMBIARAAA
    auto var_prom_independientes = prom_data(0,4);
    //SE escalan los datos
    auto datos_escalados = matData.rowwise()-matData.colwise().mean();
    //Se extrae la desviacion estandar de los datos escalados
    auto desv_stand = ExData.DevStand(datos_escalados);
    //Se ecxtraen los valores de la varieble independiente
    auto var_desv_independientes = desv_stand(0,4);
    //SE crea una matriz para almacenar los valores estimados de entrenamiento
    Eigen::MatrixXd y_train_hat = (X_train*thetasOut* var_desv_independientes).array() + var_prom_independientes;
    //Matriz para los valores reales de y
    Eigen::MatrixXd y = matData.col(4).topRows(7090);
    //SE revisa que tan bueno fue el modelo atravez de la metrica de rendimiento
    float metrica_R2 = modeloLR.R2_Score(y, y_train_hat);
    std::cout<< "Metrica R2 de entrenamiento: "<<metrica_R2<<std::endl;

    //----------------------------------------------------------------


    return EXIT_SUCCESS;
}
