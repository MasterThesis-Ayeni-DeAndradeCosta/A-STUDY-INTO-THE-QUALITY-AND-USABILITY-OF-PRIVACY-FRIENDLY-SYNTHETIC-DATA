/*
import org.yaml.snakeyaml.Yaml;
import java.io.InputStream;
import java.util.Map;

public class ConfigLoader {
    public static Map<String, Object> loadYamlConfig(String filePath) throws Exception {
        Yaml yaml = new Yaml();
        try (InputStream inputStream = ConfigLoader.class.getClassLoader().getResourceAsStream(filePath)) {
            if (inputStream == null) {
                throw new RuntimeException("could not find file: " + filePath);
            }
            return yaml.load(inputStream);
        }
    }
}
*/

import org.yaml.snakeyaml.Yaml;
import java.io.InputStream;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.util.Map;

public class ConfigLoader {
    public static Map<String, Object> loadYamlConfig() {
        try {
            // Load config.yaml using absolute path
            String configPath = Paths.get("src", "config.yaml").toAbsolutePath().toString();
            System.out.println("🔍 Looking for config.yaml at: " + configPath);

            InputStream inputStream = Files.newInputStream(Paths.get(configPath));
            Yaml yaml = new Yaml();
            return yaml.load(inputStream);
        } catch (Exception e) {
            throw new RuntimeException("could not find file: config.yaml", e);
        }
    }
}
