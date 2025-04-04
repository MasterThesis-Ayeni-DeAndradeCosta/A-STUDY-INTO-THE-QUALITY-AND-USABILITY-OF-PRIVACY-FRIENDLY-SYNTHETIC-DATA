import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.Data;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.HashMap;


//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws Exception {
        // ✅ Load the correct config file (from the arguments)
        //String configPath = "configs/benchmark_config.yaml";
        String configPath;
        if (args.length > 2) {
            configPath = new File(args[2]).getAbsolutePath();
            System.out.println("📄 Java received config path: " + configPath);
        } else {
            configPath = new File("configs/benchmark_config.yaml").getAbsolutePath();
            System.out.println("⚠️ No config path provided. Falling back to: " + configPath);
        }
        File configFile = new File(configPath);
        String absoluteConfigPath = configFile.getAbsolutePath();
        System.out.println("🔹 Loading config from: " + absoluteConfigPath); // Debugging
        Map<String, Object> yamlConfig = ConfigLoader.loadYamlConfig(absoluteConfigPath);

        // ✅ Extract dataset settings
        Map<String, Object> datasetConfig = (Map<String, Object>) yamlConfig.get("dataset");
        //String datasetPath = new File((String) datasetConfig.get("path")).getAbsolutePath();

        String datasetPath;
        if (args.length > 0) {
            datasetPath = new File(args[0]).getAbsolutePath();
            System.out.println("📂 Dataset Path (from args): " + datasetPath);
        } else
         {
            datasetPath = new File((String) datasetConfig.get("path")).getAbsolutePath();
            System.out.println("📂 Dataset Path (from config): " + datasetPath);
         }



        char datasetSeparator = datasetConfig.get("separator").toString().charAt(0); // Ensure correct delimiter
        //System.out.println("📂 Dataset Path: " + datasetPath); // Debugging
        System.out.println("🔹 Dataset Separator: " + datasetSeparator); // Debugging

        // ✅ Load dataset into ARX
        Data data = Data.create(datasetPath, StandardCharsets.UTF_8, datasetSeparator);

        // ✅ Extract anonymization settings
        Map<String, Object> anonymizationConfig = (Map<String, Object>) yamlConfig.get("anonymization");
        double suppressionLimit = (double) anonymizationConfig.get("suppression_limit");
        System.out.println("🔹 Suppression Limit: " + suppressionLimit); // Debugging

        // ✅ Extract attributes from anonymization settings
        Map<String, List<String>> attributes = (Map<String, List<String>>) anonymizationConfig.get("attributes");
        List<String> quasiIdentifiers = attributes.get("quasi_identifiers");
        List<String> sensitiveAttributes = attributes.get("sensitive_attributes");
        List<String> insensitiveAttributes = attributes.get("insensitive_attributes");

        System.out.println("\n🔹 Configured Attributes:"); // Debugging
        for (String attr : quasiIdentifiers) System.out.println("  ✅ Quasi-Identifier: " + attr);
        for (String attr : sensitiveAttributes) System.out.println("  ✅ Sensitive Attribute: " + attr);
        for (String attr : insensitiveAttributes) System.out.println("  ✅ Insensitive Attribute: " + attr);

        // ✅ Extract hierarchies correctly
        Map<String, Object> hierarchies = (Map<String, Object>) anonymizationConfig.get("hierarchies");
        Map<String, Object> hierarchyConfig = (Map<String, Object>) anonymizationConfig.get("hierarchy_config");

        // ✅ Ensure hierarchy paths are absolute
        Map<String, Object> updatedHierarchies = new HashMap<>();
        for (Map.Entry<String, Object> entry : hierarchies.entrySet()) {
            String absolutePath = new File((String) entry.getValue()).getAbsolutePath();
            updatedHierarchies.put(entry.getKey(), absolutePath);
        }

        System.out.println("\n🔹 Defined Hierarchies:"); // Debugging
        for (String attribute : updatedHierarchies.keySet()) {
            System.out.println("  ✅ " + attribute + " -> " + updatedHierarchies.get(attribute));
        }

        // ✅ Instantiate the AnonymizationManager
        AnonymizationManager anonymizationManager = new AnonymizationManager(data, updatedHierarchies, hierarchyConfig);
        anonymizationManager.configureAttributes(quasiIdentifiers, sensitiveAttributes, insensitiveAttributes);
        anonymizationManager.configureHierarchies();

        // ✅ Extract privacy models (k-Anonymity, l-Diversity)
        AnonymizationModel anonymizationModel = new AnonymizationModel();
        anonymizationModel.addModelsFromConfig((Map<String, Object>) anonymizationConfig.get("models"));

        // ✅ Build ARX anonymization configuration
        ARXConfiguration config = anonymizationModel.buildConfiguration(suppressionLimit);
        System.out.println("🔹 Privacy Models Configured"); // Debugging

        // ✅ Extract anonymized output path correctly
        //String anonymizedOutputPath = new File((String) anonymizationConfig.get("anonymized_output")).getAbsolutePath();
        String anonymizedOutputPath;
        if (args.length > 1) {
            anonymizedOutputPath = new File(args[1]).getAbsolutePath();
            System.out.println("📂 Anonymized Output Path (from args): " + anonymizedOutputPath);
        } else {
            throw new IllegalArgumentException("❌ Missing anonymized output path as second argument.");
        }

        System.out.println("📂 Anonymized Output Path: " + anonymizedOutputPath); // Debugging

        // ✅ Run the anonymization process
        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXResult result = anonymizer.anonymize(data, config);

        // ✅ Save the anonymized output
        result.getOutput(false).save(anonymizedOutputPath, datasetSeparator);
        System.out.println("✅ Anonymization completed. Output saved to: " + anonymizedOutputPath);

        // ✅ Print anonymization statistics
        DataVisualizer.printResultInfo(result);
    }
}