package org.neural_network.simple_neural_network.repository.learning_data_reposes;

import org.neural_network.simple_neural_network.tools.entity.LearningData;
import org.springframework.jdbc.core.JdbcTemplate;

import java.util.Arrays;
import java.util.List;

public abstract class LearningDataRepository {
    private final JdbcTemplate jdbcTemplate;
    private final String tableName;

    public LearningDataRepository(JdbcTemplate jdbcTemplate, String tableName) {
        this.jdbcTemplate = jdbcTemplate;
        this.tableName = tableName;
    }

    public void add(LearningData learningData) {
        String data = packData(learningData.getAnswer(), learningData.getExample());
        String sql = "insert into public.%s values(%s, '{%s}')"
                .formatted(tableName,learningData.getId(), data);
        jdbcTemplate.update(sql);
    }

    private String listToJsonMapper(List<Double> list) {
        return list.stream()
                .map(String::valueOf)
                .reduce((s1, s2) -> s1 + ", " + s2)
                .orElseThrow();
    }

    private String packData (List<Double> answer, List<Double> example) {
        String answerJson = listToJsonMapper(answer);
        String exampleJson = listToJsonMapper(example);
        return "\"" + answerJson + "\"" + ":" + "\"" + exampleJson + "\"";
    }

    public LearningData getById(int id) {
        String sql = "select * from public.%s where id = %s".formatted(tableName, id);
        return jdbcTemplate.query(sql, (rs, rowNum) -> {
            String data1 = rs.getString("data");
            List<List<Double>> lists = extractAnswerAndExample(data1);
            return new LearningData(id, lists.get(0), lists.get(1));
        }).getFirst();
    }

    private List<List<Double>> extractAnswerAndExample(String data) {
        List<String> answerAndExample = Arrays.stream(data.split(":"))
                .map(this::cleanSymbols)
                .toList();
        return List.of(
                extractValues(answerAndExample.get(0)),
                extractValues(answerAndExample.get(1))
        );
    }

    private List<Double> extractValues(String data) {
        return Arrays.stream(data.split(","))
                .map(Double::valueOf)
                .toList();

    }

    private String cleanSymbols(String data) {
        return data.replaceAll("\\s", "")
                .replaceAll("\\{", "")
                .replaceAll("\\}", "")
                .replaceAll("^\"|\"$", "");
    }

    public boolean hasNext(int id) {
        String sql = "select max(id) from public.%s".formatted(tableName);
        Integer maxId = jdbcTemplate
                .query(sql, (rs, rowNum) -> rs.getInt("max"))
                .getFirst();
        return id <= maxId;
    }

}
