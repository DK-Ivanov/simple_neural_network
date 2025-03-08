package org.neural_network.simple_neural_network.repository;

import lombok.RequiredArgsConstructor;
import org.neural_network.simple_neural_network.entity.Neuron;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class NeuronRepository {

    private final JdbcTemplate jdbcTemplate;
    private final TransactionTemplate transactionTemplate;
    private final WeightRepository weightRepository;

    public void add(Neuron neuron) {
        String sql = "insert into public.neuron values(?, ?)";
        transactionTemplate.executeWithoutResult(transactionStatus -> {
            jdbcTemplate.update(sql, ps -> {
                ps.setInt(1, neuron.getId());
                ps.setDouble(2, neuron.getBias());
            });
            neuron.getWeight()
                    .forEach(weightRepository::add);
        });
    }

    public void update(Neuron neuron) {
        transactionTemplate.executeWithoutResult(transactionStatus -> {
            deleteById(neuron.getId());
            add(neuron);
        });
    }

    public void deleteById(int id) {
        String sql = "delete from public.neuron where id = ?";
        transactionTemplate.executeWithoutResult(transactionStatus -> {
            weightRepository.deleteByNeuronId(id);
            jdbcTemplate.update(sql, ps -> ps.setInt(1, id));
        });
    }

    public Neuron getWeightsAndBias(Neuron neuron) {
        String sql = "select bias from public.neuron where id = ?";
        neuron.setBias(Optional.of(jdbcTemplate.query(sql,
                ps -> ps.setInt(1, neuron.getId()),
                (rs, rowNum) -> rs.getDouble("bias")).getFirst())
                .orElseThrow(RuntimeException::new));
        neuron.setWeight(weightRepository.get(neuron.getId()));
        return neuron;
    }

    public void clean() {
        String sqlDrop = "Drop table neuron;";
        String sqlCreate = "Create table neuron(id int primary key, bias numeric(20, 14));";
        transactionTemplate.executeWithoutResult(transactionStatus -> {
            weightRepository.dropTable();
            jdbcTemplate.execute(sqlDrop);
            jdbcTemplate.execute(sqlCreate);
            weightRepository.createTable();
        });
    }



}
