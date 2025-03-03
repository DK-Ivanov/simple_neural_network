package org.neural_network.simple_neural_network.service;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.springframework.stereotype.Service;


import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
public class DiagramService {

    private SwingWrapper<XYChart> sw;
    private XYChart chart;

    private List<BigDecimal> losses = new ArrayList<>();
    private List<Double> iterations = new ArrayList<>();
    private Double iteration = 0d;

    public void printLossCurve(BigDecimal loss) {
        iteration++;
        iterations.add(iteration);
        losses.add(loss);
        repaint();
        try {
            Thread.sleep(1);
            } catch (Exception e) {}
    }
    @PostConstruct
    private void init() {
        chart = QuickChart
                .getChart("Loss curve",
                        "Iterations",
                        "Loss",
                        "loss curve",
                        List.of(0),
                        List.of(0));
        sw = new SwingWrapper<>(chart);
        sw.setTitle("Neural Network loss curve");
        sw.displayChart();
    }

    private void repaint() {
        chart.updateXYSeries("loss curve", iterations, losses, null);
        sw.repaintChart();
    }

//    private Double round(Double loss) {
//        try {
//            BigDecimal rounder = new BigDecimal(loss);
//            rounder = rounder.setScale(10, RoundingMode.HALF_UP);
//            return rounder.doubleValue();
//        } catch (Exception e) {
//            log.warn("Нестандартная ошибка: loss = {}", loss);
//            return 0d;
//        }
//    }


}