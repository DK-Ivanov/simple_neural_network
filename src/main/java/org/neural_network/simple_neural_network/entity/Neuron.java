package org.neural_network.simple_neural_network.entity;

import lombok.Data;
import org.neural_network.simple_neural_network.tools.MathFunc;
import org.neural_network.simple_neural_network.tools.activate_function.SimpleActivateFunc;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Data
public class Neuron implements Comparable<Neuron>{
    private final int id;
    private List<Weight> weight;
    private Double bias = 0d;
    private Double lastLinearRegression;
    private int numberInLayer;
    private final SimpleActivateFunc activateFunc;

    public Neuron(int numberInLayer, int featuresCount, int id, SimpleActivateFunc activateFunc) {
        AtomicInteger weightNumber = new AtomicInteger(0);
        this.id = id;
        this.weight = Stream.generate(() -> new Weight(
                        id,
                        weightNumber.getAndAdd(1),
                        activateFunc.equals(SimpleActivateFunc.RELU) ?
                                new Random().nextDouble(0,2d/featuresCount) :
                                new Random().nextDouble(0.1, 1d)
                    )
                )
                .limit(featuresCount)
                .collect(Collectors.toList());
        this.numberInLayer = numberInLayer;
        this.activateFunc = activateFunc;
    }

    /**
     * Добавляет нелинейность в предсказание:
     * Передает выходное значение из линейной регрессии,
     * как входное значение, в функцию активации.
     * Возвращает предсказание нейрона.
     */
    public Double getSimplePredication(List<Double> features) {
        return activateFunc.getActivateFunction().activate(linearRegression(features));
    }

    /**
     * Добавляет нелинейность в предсказание:
     * Softmax(z)
     * Возвращает предсказание нейрона.
     */
    public Double getSoftmaxPredication(List<Double> otherNeuronsZ, Double maxZ) {
        return MathFunc.softmax(this.lastLinearRegression, otherNeuronsZ, maxZ);
    }

    /**
     * Суммирует все выходные значения с предыдущего слоя,
     * умноженные на соответствующие веса, и добавляет смещение.
     * Возвращает линейное предсказание нейрона.
     */
    public Double linearRegression(List<Double> features) {
        Double lastLinearRegression = (features.stream()
                .map(feature -> feature * weight.get(features.indexOf(feature)).getValue())
                .reduce(Double::sum)
                .orElseThrow(RuntimeException::new)) + bias;
        this.lastLinearRegression = lastLinearRegression;
        return lastLinearRegression;
    }

    @Override
    public int compareTo(Neuron o) {
        if (o.getNumberInLayer() > getNumberInLayer()) {
            return -1;
        }
        return 1;
    }
}
