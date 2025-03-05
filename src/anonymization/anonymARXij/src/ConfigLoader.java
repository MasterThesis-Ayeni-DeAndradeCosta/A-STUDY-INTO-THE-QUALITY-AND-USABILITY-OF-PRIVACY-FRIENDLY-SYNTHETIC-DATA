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
