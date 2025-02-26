package org.neural_network.simple_neural_network.tools.activate_function;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.neural_network.simple_neural_network.tools.MathFunc;

import java.util.stream.Stream;

@Getter
@AllArgsConstructor
public enum SimpleActivateFunc {
    NO_ACTIVATE_FUNC(0, MathFunc::noActivateFunc, MathFunc::noActivateFuncDerivative),
    SIGMOID(1, MathFunc::sigmoid, MathFunc::sigmaDerivative),
    RELU(2, MathFunc::RELU, MathFunc::ReLuDerivative),
    TANH(3, Math::tanh, MathFunc::tanhDerivative),
    SOFTMAX(4, MathFunc::fakeSoftmax, MathFunc::fakeSoftmax);
    final int number;
    final ActivateFunction activateFunction;
    final ActivateFunction activateFunctionDerivative;

    public ActivateFunction getActivateFunction() {
        return activateFunction;
    }

    public ActivateFunction getActivateFunctionDerivative() {
        return activateFunctionDerivative;
    }

    public static SimpleActivateFunc getByNumber(int number) {
        return Stream.of(SimpleActivateFunc.NO_ACTIVATE_FUNC,
                        SimpleActivateFunc.SIGMOID,
                        SimpleActivateFunc.RELU,
                        SimpleActivateFunc.TANH,
                        SimpleActivateFunc.SOFTMAX
                )
                .filter(activateFunc -> activateFunc.number == number)
                .findFirst()
                .orElseThrow();
    }



}
