package org.neural_network.simple_neural_network.repository;

import lombok.RequiredArgsConstructor;
import org.neural_network.simple_neural_network.entity.Weight;
import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class WeightRepository {

    private final JdbcTemplate jdbcTemplate;

    public void add(Weight weight) {
        String sql = "insert into public.weight values(?, ?, ?);";
        jdbcTemplate.update(sql, ps -> {
            ps.setInt(1, weight.getNEURON_ID());
            ps.setInt(2, weight.getNumberInNeuron());
            ps.setDouble(3, weight.getValue());
        });
    }

    public void deleteByNeuronId(int neuronId) {
        String sql = "delete from public.weight where neuron_id = ?";
        jdbcTemplate.update(sql, ps -> ps.setInt(1, neuronId));
    }

    public List<Weight> get(int neuronId) {
        String sql = "select * from public.weight where neuron_id = ?;";
        return jdbcTemplate.query(
                sql,
                ps -> ps.setInt(1, neuronId),
                (rs, rowNum) -> {
                    int numberInNeuron = rs.getInt("number_in_neuron");
                    int neuron_Id = rs.getInt("neuron_id");
                    double value = rs.getDouble("value");
                    return new Weight(neuron_Id, numberInNeuron, value);
                });
    }

    public void createTable() {
        String sql = "Create table weight(neuron_id int references neuron(id), number_in_neuron int, value numeric(20, 14)); ";
        jdbcTemplate.execute(sql);
    }

    public void dropTable() {
        String sql = "Drop table weight;";
        jdbcTemplate.execute(sql);
    }
}
