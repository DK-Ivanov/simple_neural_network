package org.neural_network.simple_neural_network.tools.loss;

@FunctionalInterface
public interface LossFunction {
    Double calcLoss(Double label, Double predicationValue);
}
