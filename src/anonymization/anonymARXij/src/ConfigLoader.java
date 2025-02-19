import org.json.JSONObject;

import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

public class ConfigLoader {
    public static JSONObject loadConfig(String path) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(path)), Charset.forName("UTF-8"));

        return new JSONObject(content);
    }
}
