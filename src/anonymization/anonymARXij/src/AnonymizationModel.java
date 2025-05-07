import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.Data;
// Importação correta!
import org.deidentifier.arx.criteria.*;
import org.deidentifier.arx.DataSubset;

import java.util.HashMap;
import java.util.Map;

/**
 * The AnonymizationModel class is responsible for:
 * - Defining and storing various privacy models for anonymization.
 * - Dynamically configuring anonymization techniques based on external configurations.
 * - Creating an ARXConfiguration that specifies privacy constraints.
 * - Supporting multiple privacy models like k-Anonymity, l-Diversity, and t-Closeness.
 */
public class AnonymizationModel {
    // A HashMap to dynamically store different privacy models
    // Key: Privacy model name ("k-Anonymity", "L-Diversity", etc.)
    // Value: Corresponding ARX PrivacyCriterion object
    private final Map<String, PrivacyCriterion> privacyModels = new HashMap<>();

    // Adicionar modelos de anonimização dinamicamente
    public void addKAnonymity(int k) {
        privacyModels.put("k-Anonymity", new KAnonymity(k));
    }

    public void addLDiversity(String attribute, int l) {
        privacyModels.put("L-Diversity", new DistinctLDiversity(attribute, l));
    }

    public void addEntropyLDiversity(String attribute, int l) {
        privacyModels.put("Entropy L-Diversity", new EntropyLDiversity(attribute, l));
    }

    public void addTCloseness(String attribute, double t) {
        privacyModels.put("T-Closeness", new EqualDistanceTCloseness(attribute, t));
    }

    /*
    // Retorna a configuração com os modelos aplicados
    public ARXConfiguration buildConfiguration(double suppressionLimit) {
        ARXConfiguration config = ARXConfiguration.create();
        config.setSuppressionLimit(suppressionLimit);

        for (PrivacyCriterion model : privacyModels.values()) {
            config.addPrivacyModel(model);
        }

        return config;
    }
    */


    public ARXConfiguration buildConfiguration(double suppressionLimit) {
        ARXConfiguration config = ARXConfiguration.create();
        config.setSuppressionLimit(suppressionLimit);
        privacyModels.values().forEach(config::addPrivacyModel);
        return config;
    }

    public void addModelsFromConfig(Map<String, Object> config) {
        if (config.containsKey("k_anonymity")) {
            addKAnonymity((int) config.get("k_anonymity"));
        }

        // if (config.containsKey("l_diversity")) {
        //     Map<String, Object> lDiv = (Map<String, Object>) config.get("l_diversity");
        //     addLDiversity((String) lDiv.get("attribute"), (int) lDiv.get("value"));
        // }
        if (config.containsKey("l_diversity")) {
            Map<String, Object> lDiv = (Map<String, Object>) config.get("l_diversity");
            String attribute = (String) lDiv.get("attribute");
            int value = (int) lDiv.get("value");
            String model = (String) lDiv.getOrDefault("model", "distinct");

            if ("entropy".equalsIgnoreCase(model)) {
                addEntropyLDiversity(attribute, value);
                System.out.printf("✓ Applied Entropy L-Diversity on: %s (l=%d)%n", attribute, value);
            } else {
                addLDiversity(attribute, value);
                System.out.printf("✓ Applied Distinct L-Diversity on: %s (l=%d)%n", attribute, value);
            }
        }


        if (config.containsKey("t_closeness")) {
            Map<String, Object> tClose = (Map<String, Object>) config.get("t_closeness");
            addTCloseness((String) tClose.get("attribute"), (double) tClose.get("value"));
        }
    }

    public void printSelectedModels() {
        System.out.println("Selected Anonymization Models:");
        for (String key : privacyModels.keySet()) {
            System.out.println(" - " + key);
        }
    }
}

