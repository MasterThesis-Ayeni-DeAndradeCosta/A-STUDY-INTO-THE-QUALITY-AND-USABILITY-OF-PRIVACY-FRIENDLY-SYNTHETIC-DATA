import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.Data;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;


//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

    public static void main(String[] args) throws Exception {

        // ✅ Correct: No arguments needed
        Map<String, Object> yamlConfig = ConfigLoader.loadYamlConfig();


        String datasetPath = (String) yamlConfig.get("dataset");
        Data data = Data.create(datasetPath, StandardCharsets.UTF_8, ',');

        double suppressionLimit = (double) yamlConfig.get("suppression_limit");

        Map<String, List<String>> attributes = (Map<String, List<String>>) yamlConfig.get("attributes");
        List<String> quasiIdentifiers = attributes.get("quasi_identifiers");
        List<String> sensitiveAttributes = attributes.get("sensitive_attributes");
        List<String> insensitiveAttributes = attributes.get("insensitive_attributes");
        Map<String, Object> hierarchies = (Map<String, Object>) yamlConfig.get("hierarchies");

        Map<String, Object> hierarchyConfig = (Map<String, Object>) yamlConfig.get("hierarchy_config");

        suppressionLimit = (double) yamlConfig.get("suppression_limit");


        AnonymizationManager anonymizationManager = new AnonymizationManager(data, hierarchies, hierarchyConfig);

        anonymizationManager.configureAttributes(quasiIdentifiers, sensitiveAttributes, insensitiveAttributes);

        anonymizationManager.configureHierarchies();

        AnonymizationModel anonymizationModel = new AnonymizationModel();
        anonymizationModel.addModelsFromConfig((Map<String, Object>) yamlConfig.get("models"));
        ARXConfiguration config = anonymizationModel.buildConfiguration(suppressionLimit);

        //debug
        System.out.println("Hierarquias definidas:");
        for (String attribute : hierarchies.keySet()) {
            System.out.println("Atributo: " + attribute + " | Hierarquia: " + hierarchies.get(attribute));
        }


        System.out.println("Atributos configurados:");
        for (String attr : quasiIdentifiers) {
            System.out.println("Quasi-Identifier: " + attr);
        }
        for (String attr : sensitiveAttributes) {
            System.out.println("Sensitive: " + attr);
        }
        for (String attr : insensitiveAttributes) {
            System.out.println("Insensitive: " + attr);
        }


        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXResult result = anonymizer.anonymize(data, config);

        DataVisualizer.visualizeData(data);

        System.out.print(" - Writing anonymized data to file...");

        String anonymizedOutputPath = (String) yamlConfig.get("anonymized_output");
        result.getOutput(false).save(anonymizedOutputPath, ';');
        System.out.println("Done!");

        DataVisualizer.printResultInfo(result);

    }

}