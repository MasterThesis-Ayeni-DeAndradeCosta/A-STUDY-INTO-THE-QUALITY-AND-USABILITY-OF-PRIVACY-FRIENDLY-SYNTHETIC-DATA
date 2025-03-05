import org.deidentifier.arx.*;
import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.aggregates.StatisticsEquivalenceClasses;
import org.deidentifier.arx.aggregates.StatisticsQuality;
import java.io.IOException;
import java.util.Iterator;

public class DataVisualizer {

    public static void visualizeData(Data data) {
        System.out.println("Original data:");
        DataHandle handle = data.getHandle(); // Get a handle to the original data
        int rowCount = 0;

        Iterator<String[]> iterator = handle.iterator();
        while (iterator.hasNext() && rowCount < 10) { // Limit to 10 rows
            String[] row = iterator.next();
            for (String value : row) {
                System.out.print(value + "\t");
            }
            System.out.println();
            rowCount++;
        }
    }

    public static void printResultInfo(ARXResult result) throws IOException {
        DataHandle anonymizedHandle = result.getOutput(false);

        // Access quality statistics
        StatisticsQuality utilityResult = anonymizedHandle.getStatistics().getQualityStatistics();
        System.out.println("Anonymized Data Quality Metrics:");
        System.out.println(" - Ambiguity: " + utilityResult.getAmbiguity().getValue());
        System.out.println(" - AECS: " + utilityResult.getAverageClassSize().getValue());
        System.out.println(" - Discernibility: " + utilityResult.getDiscernibility().getValue());
        System.out.println(" - Granularity: " + utilityResult.getGranularity().getArithmeticMean(false));
        System.out.println(" - MSE: " + utilityResult.getAttributeLevelSquaredError().getArithmeticMean(false));
        System.out.println(" - Non-Uniform Entropy: " + utilityResult.getNonUniformEntropy().getArithmeticMean(false));
        System.out.println(" - Precision: " + utilityResult.getGeneralizationIntensity().getArithmeticMean(false));
        System.out.println(" - Record-level SE: " + utilityResult.getRecordLevelSquaredError().getValue());

        // Access equivalence class statistics
        StatisticsEquivalenceClasses stats = anonymizedHandle.getStatistics().getEquivalenceClassStatistics();
        System.out.println("Equivalence Class Statistics:");
        System.out.println(" - Number of Classes: " + stats.getNumberOfEquivalenceClasses());
        System.out.println(" - Average Class Size: " + stats.getAverageEquivalenceClassSize());

        // Calculate suppressed records
        int totalRows = anonymizedHandle.getNumRows();
        int suppressedCount = 0;

        for (int i = 0; i < totalRows; i++) {
            if (anonymizedHandle.isSuppressed(i)) {
                suppressedCount++;
            }
        }

        double suppressionPercentage = (double) suppressedCount / totalRows * 100;

        // Print suppression statistics
        System.out.println("Suppression Statistics:");
        System.out.println(" - Suppressed Records: " + suppressedCount);
        System.out.println(" - Suppression Percentage: " + String.format("%.2f", suppressionPercentage) + "%");
    }

    public static void printResult(ARXResult result, Data data) {
        System.out.println("Anonymization complete.");
        System.out.println("Optimal generalization level: " + result.getGlobalOptimum().getGeneralization("age"));

        // Print the first few rows of the anonymized data
        System.out.println("Anonymized data preview:");
        DataHandle handle = result.getOutput(false);
        int rowCount = 0;

        Iterator<String[]> iterator = handle.iterator();
        while (iterator.hasNext() && rowCount < 10) { // Limit to 10 rows for preview
            String[] row = iterator.next();
            for (String value : row) {
                System.out.print(value + "\t");
            }
            System.out.println();
            rowCount++;
        }
    }
}
