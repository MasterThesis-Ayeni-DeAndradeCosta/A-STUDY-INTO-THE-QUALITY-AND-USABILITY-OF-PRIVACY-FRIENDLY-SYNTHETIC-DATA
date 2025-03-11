
import org.deidentifier.arx.*;
import org.deidentifier.arx.aggregates.HierarchyBuilder;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import java.nio.file.Paths;
import java.io.File;

public class AnonymizationManager {
    private static Data data;
    private Map<String, Object> hierarchies;
    private Map<String, Object> hierarchyConfig;


    public AnonymizationManager(Data data, Map<String, Object> hierarchies, Map<String, Object> hierarchyConfig) {
        this.data = data;
        this.hierarchies = hierarchies;
        this.hierarchyConfig = hierarchyConfig;
    }

    public static void loadHierarchy() {
        String hierarchyPath = Paths.get("src", "hierarchies", "hierarchy_age_4.csv").toAbsolutePath().toString();
        File file = new File(hierarchyPath);

        if (!file.exists()) {
            System.out.println("❌ File Not Found: " + file.getAbsolutePath());
        } else {
            System.out.println("✅ Found: " + file.getAbsolutePath());
        }

        // Now use hierarchyPath instead of relative "src/hierarchies/hierarchy_age_4.csv"
    }


    public void setAttributeQuasiD(String attribute) {
        data.getDefinition().setAttributeType(attribute, AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
    }

    public void setAttributeSensitive(String attribute) {
        data.getDefinition().setAttributeType(attribute, AttributeType.SENSITIVE_ATTRIBUTE);
    }

    public void setAttributeInsensitive(String attribute) {
        data.getDefinition().setAttributeType(attribute, AttributeType.INSENSITIVE_ATTRIBUTE);
    }

    public void configureAttributes(List<String> quasiIdentifiers, List<String> sensitiveAttributes, List<String> insensitiveAttributes) {
        for (String attribute : quasiIdentifiers) {
            setAttributeQuasiD(attribute);
        }
        for (String attribute : sensitiveAttributes) {
            setAttributeSensitive(attribute);
        }
        for (String attribute : insensitiveAttributes) {
            setAttributeInsensitive(attribute);
        }
    }

    public void configureHierarchies() {
        for (Map.Entry<String, Object> entry : hierarchies.entrySet()) {
            String attribute = entry.getKey();
            Object hierarchyData = entry.getValue();

            try {
                if (hierarchyData instanceof String) {
                    setHierarchyToAtt(attribute, (String) hierarchyData);
                } else if (hierarchyData instanceof HierarchyBuilder<?>) {
                    setHierarchyToAtt(attribute, (HierarchyBuilder<?>) hierarchyData);
                } else {
                    System.err.println("Unknown hierarchy type for attribute: " + attribute);
                }
            } catch (IOException e) {
                System.err.println("Error in loading hierarchy to " + attribute + ": " + e.getMessage());
            }
        }

        // Adicionando hierarquia baseada em intervalo para atributos especificados
        for (Map.Entry<String, Object> entry : hierarchyConfig.entrySet()) {
            String attribute = entry.getKey();
            int numIntervals = (int) entry.getValue();

            if (!hierarchies.containsKey(attribute)) {  // Só cria se não houver hierarquia já definida
                try {
                    createIntervalHierarchy(attribute, numIntervals);
                    System.out.println("Hierarquia intervalar criada para: " + attribute);
                } catch (Exception e) {
                    System.err.println("Erro ao criar hierarquia intervalar para " + attribute + ": " + e.getMessage());
                }
            }
        }
    }

    /*
    public void configureHierarchies() {
        for (Map.Entry<String, Object> entry : hierarchies.entrySet()) {
            String attribute = entry.getKey();
            Object hierarchyData = entry.getValue();

            try {
                if (hierarchyData instanceof String) {
                    // CSV-based hierarchy
                    setHierarchyToAtt(attribute, (String) hierarchyData);
                } else if (hierarchyData instanceof HierarchyBuilder<?>) {
                    // hierarchy from builder
                    setHierarchyToAtt(attribute, (HierarchyBuilder<?>) hierarchyData);
                } else {
                    System.err.println("Unknown hierarchy type for attribute: " + attribute);
                }
            } catch (IOException e) {
                System.err.println("Error in loading hierarchy to " + attribute + ": " + e.getMessage());
            }
        }
    }

     */


    public void setHierarchyToAtt(String attributeName, Object hierarchy) throws IOException {
        if (hierarchy instanceof String) {
            data.getDefinition().setAttributeType(attributeName,
                    AttributeType.Hierarchy.create((String) hierarchy, StandardCharsets.UTF_8, ';'));
        } else if (hierarchy instanceof HierarchyBuilder<?>) {
            data.getDefinition().setAttributeType(attributeName, (HierarchyBuilder<?>) hierarchy);
        } else {
            throw new IllegalArgumentException("Formato de hierarquia inválido para atributo: " + attributeName);
        }
    }

    public void createIntervalHierarchy(String attributeName, int numIntervals) throws IOException {
        DataHandle handle = data.getHandle();
        int columnIndex = handle.getColumnIndexOf(attributeName);
        if (columnIndex == -1) {
            throw new IllegalArgumentException("Atributo não encontrado: " + attributeName);
        }

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int i = 0; i < handle.getNumRows(); i++) {
            String value = handle.getValue(i, columnIndex);
            if (value != null && !value.isEmpty()) {
                double numValue = Double.parseDouble(value);
                if (numValue < min) min = numValue;
                if (numValue > max) max = numValue;
            }
        }

        double intervalSize = (max - min) / numIntervals;
        HierarchyBuilderIntervalBased<Double> builder = HierarchyBuilderIntervalBased.create(DataType.DECIMAL);

        double currentStart = min;
        for (int i = 0; i < numIntervals; i++) {
            double currentEnd = currentStart + intervalSize;
            if (i == numIntervals - 1) {
                currentEnd = max + 0.000001;
            }
            builder.addInterval(currentStart, currentEnd);
            currentStart = currentEnd;
        }

        builder.getLevel(0).addGroup(2);
        builder.getLevel(1).addGroup(3);

        setHierarchyToAtt(attributeName, builder);
    }
}
