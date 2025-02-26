package org.neural_network.simple_neural_network.tools;

import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.List;

@Component
public class MathFunc {

    private static Double outputCount = 2d;

    /**
     * Нет функции активации
     */
    public static Double noActivateFunc(Double z) {
        return z;
    }

    /**
     * Нет функции активации
     */
    public static Double noActivateFuncDerivative(Double z) {
        return 1d;
    }

    /**
     * Функция активации RELU
     */
    public static Double RELU(Double z) {
        return Math.max(0.0001*z, z);
    }

    /**
     * Производная функции RELU
     */
    public static Double ReLuDerivative(Double z) {
        return z > 0 ? 1d: 0d;
    }

    /**
     * Функция активации Сигмоида
     */
    public static Double sigmoid(Double z) {
        return 1d / (1d + Math.exp(-z));
    }

    /**
     * Производная сигмоиды от z
     */
    public static Double sigmaDerivative(Double z) {
        return Math.exp(-z) / (Math.pow(Math.exp(-z) + 1, 2));
    }

    /**
     * Функция активации softmax
     */
    public static Double fakeSoftmax(Double z) {
        throw new RuntimeException("Для этой функции нужна другая реализация класса NeuralNetwork");
    }
    /**
     * Функция активации softmax
     */
    public static Double softmax(Double z, List<Double> otherNeuronsZ, Double maxZ) {
        Double otherExps = otherNeuronsZ.stream().map(otherZ -> Math.exp(otherZ - maxZ)).reduce(Double::sum).orElseThrow();
        return new BigDecimal(Math.exp(z - maxZ) / otherExps).doubleValue();
    }

    /**
     * Производная функции softmax
     */
    public static Double softmaxDerivative(Double z, List<Double> otherNeuronZ, Double maxZ) {
        Double softmax = softmax(z, otherNeuronZ, maxZ);
        return softmax * (1 - softmax);
    }

    /**
     * Производная функции tanh
     */
    public static Double tanhDerivative(Double z) {
        return 1d - Math.pow(Math.tanh(z), 2);
    }

    /**
     * Вычисление потери MSE
     */
    public static Double MSE(Double predicatedValue, Double label) {
        return Math.pow(label - predicatedValue, 2);
    }

    /**
     * Вычисление производной потери MSE.
     */
    public static Double MSEDerivative(Double predicatedValue, Double label) {
        return 2*(label - predicatedValue);
    }

    /**
     * Вычисление потери MAE
     */
    public static Double MAE(Double predicatedValue, Double label) {
        return Math.abs(predicatedValue - label);
    }

    /**
     * Вычисление производной потери MAE.
     */
    public static Double MAEDerivative(Double predicatedValue, Double label) {
        if (label > predicatedValue) {
            return -1d;
        }
        if (predicatedValue > label) {
            return 1d;
        }
        return 0d;
    }

    /**
     * Вычисление потери Log loss
     */
    public static Double logLoss(Double predicatedValue, Double label) {
        return -predicatedValue * Math.log(label + 0.000000001);
    }

    /**
     * Вычисление производной потери Log loss
     */
//    public static Double logLossDerivative(Double predicatedValue, Double label) {
//        return -predicatedValue / label;
//    }
    public static Double logLossDerivative(Double predicatedValue, Double label) {
        return label - predicatedValue;
    }

    /**
     * Вычисление производной потери Log loss для функции активации softmax
     */
    public static Double softmaxLogLossDerivative(Double predicatedValue, Double label) {
        return label - predicatedValue;
    }

}
