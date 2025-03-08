package org.neural_network.simple_neural_network.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;

@Data
@AllArgsConstructor
public class Weight implements Comparable<Weight> {
    private final int NEURON_ID;
    private int numberInNeuron;
    private Double value = 0d;

    public Weight setValueAndGetWeight(Double value) {
        this.value = value;
        return this;
    }

    @Override
    public String toString() {
        return "Weight{" +
                "value=" + value +
                '}';
    }

    @Override
    public int compareTo(Weight o) {
        if (o.getNumberInNeuron() > getNumberInNeuron()) {
            return -1;
        }
        return 1;
    }
}
