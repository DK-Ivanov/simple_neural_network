package org.neural_network.simple_neural_network.tools.loss;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.neural_network.simple_neural_network.tools.MathFunc;

import java.util.stream.Stream;

@AllArgsConstructor
@Getter
public enum LossType {
    MSE(1, MathFunc::MSE, MathFunc::MSEDerivative),
    MAE(2, MathFunc::MAE, MathFunc::MAEDerivative),
    LOG_LOSS(3, MathFunc::logLoss, MathFunc::logLossDerivative);

    final int number;
    final LossFunction lossFunction;
    final LossFunctionDerivative lossFunctionDerivative;

    public static LossType getByNumber(int number) {
        return Stream.of(LossType.MSE,
                        LossType.MAE,
                        LossType.LOG_LOSS)
                .filter(lossType -> lossType.number == number)
                .findFirst()
                .orElseThrow();
    }
}
