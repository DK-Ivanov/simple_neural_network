package org.neural_network.simple_neural_network.tools.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
public class LearningData {
    private static int idCounter = 0;

    private int id;
    private List<Double> answer;
    private List<Double> example;

    public LearningData(List<Double> answer, List<Double> example) {
        this.answer = answer;
        this.example = example;
        this.id = idCounter;
        idCounter++;
    }

    public LearningData(int id, List<Double> answer, List<Double> example) {
        this.id = id;
        this.answer = answer;
        this.example = example;
    }
}
