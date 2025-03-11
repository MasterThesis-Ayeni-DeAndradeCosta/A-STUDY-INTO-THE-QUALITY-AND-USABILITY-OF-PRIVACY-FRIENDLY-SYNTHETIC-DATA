import org.yaml.snakeyaml.Yaml;
import java.io.InputStream;
import java.util.Map;
import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Paths;
import java.nio.file.Files;


// public class ConfigLoader {
//     public static Map<String, Object> loadYamlConfig(String filePath) throws Exception {
//         Yaml yaml = new Yaml();
//         try (InputStream inputStream = ConfigLoader.class.getClassLoader().getResourceAsStream(filePath)) {
//             if (inputStream == null) {
//                 throw new RuntimeException("could not find file: " + filePath);
//             }
//             return yaml.load(inputStream);
//         }
//     }
// }

public class ConfigLoader {
    public static Map<String, Object> loadYamlConfig(String filePath) throws Exception {
        Yaml yaml = new Yaml();

        // Convert relative path to absolute path
        String absolutePath = Paths.get(filePath).toAbsolutePath().toString();
        File configFile = new File(absolutePath);

        if (!configFile.exists()) {
            throw new RuntimeException("Could not find file: " + absolutePath);
        }

        System.out.println("Loading config from: " + absolutePath); // Debugging print

        try (InputStream inputStream = new FileInputStream(configFile)) {
            return yaml.load(inputStream);
        }
    }
}