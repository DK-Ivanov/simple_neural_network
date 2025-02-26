package org.neural_network.simple_neural_network.tools.exception;

public class PredictionException extends RuntimeException {
    public PredictionException() {
        super("Ошибка: предсказание равно null.");
    }
}
