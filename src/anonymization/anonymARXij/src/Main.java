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
        // Load the YAML configuration file into a Map (key-value pairs)
        //Map<String, Object> yamlConfig = ConfigLoader.loadYamlConfig("config.yaml");
        //Map<String, Object> yamlConfig = ConfigLoader.loadYamlConfig("src/anonymization/anonymARXij/src/config.yaml");
        // Dynamically get the relative path of the config file
        String configPath = "src/anonymization/anonymARXij/src/config.yaml";
        File configFile = new File(configPath);
        String absoluteConfigPath = configFile.getAbsolutePath();
        Map<String, Object> yamlConfig = ConfigLoader.loadYamlConfig(absoluteConfigPath);

        //String datasetPath = (String) yamlConfig.get("dataset");
        String datasetPath = new File((String) yamlConfig.get("dataset")).getAbsolutePath();
        
        // Load the dataset into the ARX framework using UTF-8 encoding and a comma (,) as the delimiter
        Data data = Data.create(datasetPath, StandardCharsets.UTF_8, ','); // potential problem identified, the dataset uses ; not ,
        // Retrieve the suppression limit (the max percentage of records that can be suppressed)
        double suppressionLimit = (double) yamlConfig.get("suppression_limit");
        // Extract attribute classification from the configuration file
        Map<String, List<String>> attributes = (Map<String, List<String>>) yamlConfig.get("attributes");

        // Extract lists of quasi-identifiers, sensitive attributes, and insensitive attributes
        List<String> quasiIdentifiers = attributes.get("quasi_identifiers");
        List<String> sensitiveAttributes = attributes.get("sensitive_attributes");
        List<String> insensitiveAttributes = attributes.get("insensitive_attributes");

        // Retrieve predefined hierarchies from the configuration file
        Map<String, Object> hierarchies = (Map<String, Object>) yamlConfig.get("hierarchies");

        //added code Dele
        Map<String, Object> updatedHierarchies = new HashMap<>();
        for (Map.Entry<String, Object> entry : hierarchies.entrySet()) {
            String absolutePath = new File((String) entry.getValue()).getAbsolutePath();
            updatedHierarchies.put(entry.getKey(), absolutePath);
        }


        // Retrieve hierarchy configuration settings (e.g., for numeric interval-based hierarchies)
        Map<String, Object> hierarchyConfig = (Map<String, Object>) yamlConfig.get("hierarchy_config");

        suppressionLimit = (double) yamlConfig.get("suppression_limit");

        // Instantiate the AnonymizationManager to handle attribute types and hierarchies
        //AnonymizationManager anonymizationManager = new AnonymizationManager(data, hierarchies, hierarchyConfig);
        //use updatedHierarchies instead of hierarchies
        AnonymizationManager anonymizationManager = new AnonymizationManager(data, updatedHierarchies, hierarchyConfig);

        // Configure the dataset attributes: Define which columns are quasi-identifiers, sensitive, or insensitive
        anonymizationManager.configureAttributes(quasiIdentifiers, sensitiveAttributes, insensitiveAttributes);

        // Apply hierarchy configurations (either predefined or dynamically generated)
        anonymizationManager.configureHierarchies();

        // Instantiate the AnonymizationModel to define privacy constraints
        AnonymizationModel anonymizationModel = new AnonymizationModel();

        // Load privacy models (e.g., k-Anonymity, l-Diversity) from configuration
        anonymizationModel.addModelsFromConfig((Map<String, Object>) yamlConfig.get("models"));

        // Build the final ARXConfiguration with the specified suppression limit
        ARXConfiguration config = anonymizationModel.buildConfiguration(suppressionLimit);

        //debug
        System.out.println("Defined hierarchies:");
        for (String attribute : hierarchies.keySet()) {
            System.out.println("Attribute: " + attribute + " | Hierarchy: " + hierarchies.get(attribute));
        }

        System.out.println("\nConfigured attributes:");
        for (String attr : quasiIdentifiers) {
            System.out.println("Quasi-Identifier: " + attr);
        }
        for (String attr : sensitiveAttributes) {
             System.out.println("Sensitive Attribute: " + attr);
        }
        for (String attr : insensitiveAttributes) {
            System.out.println("Insensitive Attribute: " + attr);
        }
        // Create an instance of ARXAnonymizer to perform the anonymization
        ARXAnonymizer anonymizer = new ARXAnonymizer();

        // Execute the anonymization process using the dataset and privacy configuration
        ARXResult result = anonymizer.anonymize(data, config);

        // Display a preview of the original dataset before anonymization
        DataVisualizer.visualizeData(data);
        System.out.print(" - Writing anonymized data to file...");
        // Retrieve the output file path from the configuration
        String anonymizedOutputPath = (String) yamlConfig.get("anonymized_output");
        // Save the anonymized dataset with a semicolon (;) delimiter
        result.getOutput(false).save(anonymizedOutputPath, ';');
        System.out.println("Done!");
        // Print anonymization statistics (suppression rate, equivalence class sizes, etc.)
        DataVisualizer.printResultInfo(result);
    }

}