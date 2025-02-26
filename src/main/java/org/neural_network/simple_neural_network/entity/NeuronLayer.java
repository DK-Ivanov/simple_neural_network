package org.neural_network.simple_neural_network.entity;

import lombok.Data;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import java.util.List;
import java.util.stream.Collectors;

@Data
@RequiredArgsConstructor
public class NeuronLayer implements Comparable<NeuronLayer> {

    /**
     * Передаю при создании слоя
     */
    private final int LAYER_NUMBER;
    @NonNull private List<Neuron> neurons;
    private final int inputDataCount;

    /**
     * Передаю через сеттеры после создания всех слоев
     */
    private NeuronLayer next;
    private NeuronLayer previous;

    /**
     * Обновляю при каждом обучении
     */
    private List<Double> lastInputData;
    private List<Double> lastLossDerivativeByOutputData;

    /**
     * Проверка, является ли данный слой последним.
     */
    public boolean hasNext() {
        if (this.next == null) {
            return false;
        }
        return true;
    }

    /**
     * Принимает входные параметры.
     * Возвращает список предсказаний слоя.
     */
    public List<Double> doPredication(List<Double> featuresVector) {
        this.lastInputData = featuresVector;
        return neurons.parallelStream()
                .map(neuron -> neuron.getSimplePredication(featuresVector))
                .collect(Collectors.toList());
    }

    /**
     * Вычисляю производную потерь по выходным данным для этого слоя.
     * Для последнего слоя это вычисляется в методе единичной тренировки модели.
     */
    public void calcLossDerivativeByOutput() {
        if (!hasNext()) {
            throw new RuntimeException("Сюда не должен заходить выходной слой!");
        }
        List<Double> weightsMultiplyLossDerivativeForEachNeuron = neurons.stream()
                .map(neuron -> getWeightsMultiplyLossDerivativeForEachNeuron(
                        neuron,
                        next.getLastLossDerivativeByOutputData()))
                .toList();
        this.lastLossDerivativeByOutputData = neurons.stream()
                .map(neuron ->
                        neuron.getActivateFunc().getActivateFunctionDerivative().activate(neuron.getLastLinearRegression()) *
                        weightsMultiplyLossDerivativeForEachNeuron.get(neuron.getNumberInLayer())
                )
                .toList();
    }

    /**
     * Итерируюсь по нейронам следующего слоя, достаю из них веса,
     * связанные с переданным нейроном этого слоя и домножаю каждый
     * из этих весов на соответсвующую производную ошибки по выходному значению следующего слоя.
     */
    private Double getWeightsMultiplyLossDerivativeForEachNeuron(
            Neuron neuron,
            List<Double> lossDerivative) {
        return next.getNeurons().parallelStream()
                .map(neuronFromNextLayer -> neuronFromNextLayer
                        .getWeight()
                        .get(neuron.getNumberInLayer()).getValue() *
                        lossDerivative.get(neuronFromNextLayer.getNumberInLayer())
                )
                .reduce(Double::sum)
                .orElseThrow(RuntimeException::new);
    }

    @Override
    public String toString() {
        return "NeuronLayer{" +
                "inputDataCount=" + inputDataCount +
                ", LAYER_NUMBER=" + LAYER_NUMBER +
                ", NeuronsCount=" + neurons.size() +
                '}';
    }

    @Override
    public int compareTo(NeuronLayer o) {
        if (o.getLAYER_NUMBER() > getLAYER_NUMBER()) {
            return -1;
        }
        return 1;
    }
}
