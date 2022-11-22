#include "linearregression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

//Primera funcion: FUncion de costo para la regresion lineal
// Basada en los minimos cuadrados ordinarios demostrado en clase

float LinearRegression::F_OLS_Costo(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd thetas)
{
    Eigen::MatrixXd m_interior = (pow((X*thetas - y).array(),2));
    return  (m_interior.sum())/(2*X.rows());
}

//Funcion de gradiente descendiente: EN funcion de un ratio de aprendizaje
// se avanza hasta encontrar el punto minimo que representa el valor optimo para la funcion

std::tuple<Eigen::VectorXd, std::vector<float>> LinearRegression::GradientDescent(Eigen::MatrixXd X,
                                                                                  Eigen::MatrixXd y,
                                                                                  Eigen::VectorXd thetas,
                                                                                  float alpha,
                                                                                  int num_iter){
    Eigen::MatrixXd temporal = thetas;
    int parametros = thetas.rows();
    std::vector<float> costo;
    // En costo ingresaremos los valores de la funcion de coto
    costo.push_back(F_OLS_Costo(X,y,thetas));
    //SE itera segun el numero de iteraciones y el ratio de aprendizaje para
    // encontrar los valores optimos
    for(int i=0; i<num_iter; i++){
        Eigen::MatrixXd error = X*thetas-y;
        for(int j = 0;j<parametros;j++){
            Eigen::MatrixXd X_i =X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = thetas(j,0)-(alpha/X.rows())*termino.sum();

        }
        thetas = temporal;
        // En costo ingresaremos los valores de la funcion de coto
        costo.push_back(F_OLS_Costo(X,y,thetas));
    }
    return std::make_tuple(thetas,costo);
}
/* A continuacion se presenta la funcion para revisar que tan bueno es nuestro modelo:
 * se procede a crear la materica de rendimiento:
 * crear r cuadrado score: coeficiente de determinacion, en donde
 * el mejor valor posible es 1
 */
float LinearRegression::R2_Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
     auto numerador = pow((y - y_hat).array(),2).sum();
     auto denominador = pow((y.array() - y.mean()),2).sum();

    return (1-(numerador/denominador));
}
