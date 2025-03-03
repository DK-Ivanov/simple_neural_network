package org.neural_network.simple_neural_network.tools.loss;

import java.math.BigDecimal;

@FunctionalInterface
public interface LossFunction {
    BigDecimal calcLoss(Double label, Double predicationValue);
}
