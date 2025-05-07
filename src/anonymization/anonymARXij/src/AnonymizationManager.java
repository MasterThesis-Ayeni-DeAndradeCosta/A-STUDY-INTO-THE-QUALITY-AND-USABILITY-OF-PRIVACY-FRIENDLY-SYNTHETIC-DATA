import org.deidentifier.arx.*;
import org.deidentifier.arx.aggregates.HierarchyBuilder;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased.Range;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.io.File;
import java.util.Locale;

/**
 * The AnonymizationManager class is responsible for:
 * - Assigning attributes to different privacy categories (Quasi-ID, Sensitive, Insensitive).
 * - Configuring hierarchies for generalization (both predefined CSV-based and interval-based).
 * - Ensuring attributes have correct privacy definitions before anonymization.
 */
public class AnonymizationManager {
    // Stores the dataset that will be anonymized (shared across all instances)
    private static Data data;
    // Stores predefined hierarchies (CSV-based or programmatically defined)
    private Map<String, Object> hierarchies;
    // Stores settings for numeric interval-based hierarchies
    private Map<String, Object> hierarchyConfig;

    /**
     * Constructor for the AnonymizationManager.
     * Initializes the dataset and hierarchy settings.
     *
     * @param data             The dataset to be anonymized.
     * @param hierarchies      A map containing predefined hierarchies.
     * @param hierarchyConfig  A map defining attributes that require interval-based hierarchies.
     */
    public AnonymizationManager(Data data, Map<String, Object> hierarchies, Map<String, Object> hierarchyConfig) {
        this.data = data;
        this.hierarchies = hierarchies;
        this.hierarchyConfig = hierarchyConfig;
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

    /**
     * Configures attribute hierarchies.
     * - Applies predefined CSV-based hierarchies.
     * - Generates numeric interval-based hierarchies if needed.
     */
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

        // Process interval-based hierarchies for numeric attributes
        for (Map.Entry<String, Object> entry : hierarchyConfig.entrySet()) {
            String attribute = entry.getKey();
            int numIntervals = (int) entry.getValue();

            // Only create an interval-based hierarchy if a predefined one doesn't exist
            if (!hierarchies.containsKey(attribute)) {  // Só cria se não houver hierarquia já definida
                try {
                    createIntervalHierarchy(attribute, numIntervals);
                    System.out.println("Interval-based hierarchy created for: " + attribute);
                } catch (Exception e) {
                    System.err.println("Error creating interval-based hierarchy for " + attribute + ": " + e.getMessage());
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


    // public void setHierarchyToAtt(String attributeName, Object hierarchy) throws IOException {
    //     if (hierarchy instanceof String) {
    //         // CSV-based hierarchy
    //         String filePath = (String) hierarchy;
    //         // Debugging print: Check the actual file path Java is looking for
    //         System.out.println("Looking for hierarchy file at: " + new java.io.File(filePath).getAbsolutePath());
    //         data.getDefinition().setAttributeType(attributeName,
    //                 AttributeType.Hierarchy.create((String) hierarchy, StandardCharsets.UTF_8, ';'));
    //     } else if (hierarchy instanceof HierarchyBuilder<?>) {
    //         // Programmatically built hierarchy
    //         data.getDefinition().setAttributeType(attributeName, (HierarchyBuilder<?>) hierarchy);
    //     } else {
    //         throw new IllegalArgumentException("Invalid hierarchy format for attribute: " + attributeName);
    //     }
    // }


    //new one
    public void setHierarchyToAtt(String attributeName, Object hierarchy) throws IOException {
        if (hierarchy instanceof String) {
            // Debugging: Print where Java is searching for the file
            String filePath = (String) hierarchy;
            String absolutePath = new File(filePath).getAbsolutePath();
            System.out.println("Java is searching for hierarchy file at: " + absolutePath);
    
            // Use the absolute path to avoid confusion
            data.getDefinition().setAttributeType(attributeName,
                    AttributeType.Hierarchy.create(absolutePath, StandardCharsets.UTF_8, ';'));
        } else if (hierarchy instanceof HierarchyBuilder<?>) {
            // Programmatically built hierarchy
            data.getDefinition().setAttributeType(attributeName, (HierarchyBuilder<?>) hierarchy);
        } else {
            throw new IllegalArgumentException("Invalid hierarchy format for attribute: " + attributeName);
        }
    }

    public void createIntervalHierarchy(String attributeName, int numIntervals) throws IOException {
        DataHandle handle = data.getHandle();
        int columnIndex = handle.getColumnIndexOf(attributeName);
    
        if (columnIndex == -1) {
            throw new IllegalArgumentException("Attribute not found: " + attributeName);
        }
    
        // Step 1: Compute min and max
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
    
        // Step 2: Compute interval size
        double intervalSize = (max - min) / numIntervals;
    
        // Step 3: Create hierarchy
        DataType<Double> dataType = DataType.createDecimal("#.####", Locale.ENGLISH);
    
        double lower = min;
        double upper = max;
    
        HierarchyBuilderIntervalBased<Double> builder = HierarchyBuilderIntervalBased.create(
            dataType,
            new Range<>(lower, lower, lower),
            new Range<>(upper, upper, upper)
        );
    
        builder.setAggregateFunction(dataType.createAggregate().createIntervalFunction(true, false));
        builder.addInterval(0d, intervalSize);
    
        builder.getLevel(0).addGroup(2);
        builder.getLevel(1).addGroup(3);
    
        setHierarchyToAtt(attributeName, builder);
    }
}
