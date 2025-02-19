import org.json.JSONObject;
import org.json.JSONArray;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.deidentifier.arx.*;
import org.deidentifier.arx.aggregates.*;

import java.io.IOException;
import java.nio.charset.Charset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import org.deidentifier.arx.DataType;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased.Interval;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased.Range;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.Map;

import org.deidentifier.arx.DataType;
import org.deidentifier.arx.aggregates.HierarchyBuilder.Type;
import org.deidentifier.arx.aggregates.HierarchyBuilderDate.Granularity;
import org.deidentifier.arx.aggregates.HierarchyBuilderGroupingBased.Level;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased.Interval;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased.Range;
import org.deidentifier.arx.aggregates.HierarchyBuilderPriorityBased.Priority;
import org.deidentifier.arx.aggregates.HierarchyBuilderRedactionBased.Order;

import cern.colt.Arrays;

//from example2
import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.Data.DefaultData;
import org.deidentifier.arx.criteria.EntropyLDiversity;
import org.deidentifier.arx.criteria.DistinctLDiversity;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.DataDefinition;
import org.deidentifier.arx.metric.Metric;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    private static Data data;
    private static HierarchyBuilderIntervalBased<Double> latBuilder;
    private static HierarchyBuilderIntervalBased<Double> lonBuilder;
    private static Hierarchy.DefaultHierarchy crimeCodeHierarchy;


    public static void main(String[] args) throws IOException {

        data = Data.create("C:/Users/isabe/Documents/KUL24_25/MASTER_THESIS/first_sem/datasets/crimeLAPD/Crime_Data_from_2020_to_Present.csv", StandardCharsets.UTF_8, ',');

        visualizeData(data);

        // Create a data definition
        setAttributeQuasiD("Victim_age");
        setAttributeQuasiD("Victim_sex");
        setAttributeQuasiD("Victim_descent");
        setAttributeQuasiD("LAT");
        setAttributeQuasiD("LON");

        //setAttributeSensitive("Crime_Code"); //for l-diversity

        setAttributeInsensitive("DR_NO");
        setAttributeInsensitive("Time_occured");
        setAttributeInsensitive("Premis");
        setAttributeInsensitive("Weapon");
        setAttributeInsensitive("Status");
        setAttributeInsensitive("Crime_Code");


        // Hierarchy input files
        setHierarchyToAtt("Victim_age", "src/hierarchies/hierarchy_age_4.csv");
        setHierarchyToAtt("Victim_sex", "src/hierarchies/hierarchy_sex.csv");
        setHierarchyToAtt("Victim_descent", "src/hierarchies/hierarchy_descent.csv");
        latHierarchy(10);
        setHierarchyToAtt("LAT", latBuilder);
        lonHierarchy();
        setHierarchyToAtt("LON", lonBuilder);

        //createCrimeCodeHierarchy();
        //data.getDefinition().setAttributeType("Crime_Code", crimeCodeHierarchy);


        //config arx
        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXConfiguration config = ARXConfiguration.create();

        config.addPrivacyModel(new KAnonymity(2)); // Apply 2-anonymity
        //config.addPrivacyModel(new DistinctLDiversity("Crime_Code", 2));

        config.setSuppressionLimit(1.0); // Set suppression limit to x%
        //config.setQualityModel(Metric.createEntropyMetric());

        // Execute the anonymization
        ARXResult result = anonymizer.anonymize(data, config);

        printResult(result, data);

        // Save the anonymized dataset
        System.out.print(" - Writing anonymized data to file...");
        result.getOutput(false).save("C:/Users/isabe/Documents/KUL24_25/MASTER_THESIS/first_sem/datasets/crimeLAPD/Crime_Data_anonymized.csv", ';');
        System.out.println("Done!");

        printResultInfo(result);

    }

    /**
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     **/

    private static void setAttributeQuasiD(String attribute) {
        data.getDefinition().setAttributeType(attribute, AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
    }

    private static void setAttributeSensitive(String attribute) {
        data.getDefinition().setAttributeType(attribute, AttributeType.SENSITIVE_ATTRIBUTE);
    }

    private static void setAttributeInsensitive(String attribute) {
        data.getDefinition().setAttributeType(attribute, AttributeType.INSENSITIVE_ATTRIBUTE);
    }

    private static void visualizeData(Data data) {
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

    private static void setHierarchyToAtt(String attributeName, String hierarchyFilePath) throws IOException {
        try {
            data.getDefinition().setAttributeType(attributeName,
                    Hierarchy.create(hierarchyFilePath, StandardCharsets.UTF_8, ';'));
        } catch (IOException e) {
            throw new IOException("Could not set hierarchy from csv for attribute: " + attributeName, e);
        }
    }

    private static void setHierarchyToAtt(String attributeName, HierarchyBuilder<?> builder) {
        data.getDefinition().setAttributeType(attributeName, builder);
    }


    private static void printResultInfo(ARXResult result) throws IOException {
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


    private static void printResult(ARXResult result, Data data) {
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

    private static Hierarchy.DefaultHierarchy createCrimeCodeHierarchy() {
        crimeCodeHierarchy = Hierarchy.create();

        // Theft-related crimes
        crimeCodeHierarchy.add("VEHICLE - STOLEN", "STOLEN PROPERTY", "*");
        crimeCodeHierarchy.add("BIKE - STOLEN", "STOLEN PROPERTY", "*");
        crimeCodeHierarchy.add("BURGLARY FROM VEHICLE", "PROPERTY DAMAGE", "*");
        crimeCodeHierarchy.add("THEFT OF IDENTITY", "FRAUD", "*");
        crimeCodeHierarchy.add("THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD", "FRAUD", "*");
        crimeCodeHierarchy.add("THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)", "STOLEN PROPERTY", "*");
        crimeCodeHierarchy.add("THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)", "STOLEN PROPERTY", "*");
        crimeCodeHierarchy.add("SHOPLIFTING-GRAND THEFT ($950.01 & OVER)", "STOLEN PROPERTY", "*");
        crimeCodeHierarchy.add("THEFT PLAIN - PETTY ($950 & UNDER)", "STOLEN PROPERTY", "*");
        crimeCodeHierarchy.add("BURGLARY", "PROPERTY DAMAGE", "*");

        // Assault and battery
        crimeCodeHierarchy.add("BATTERY - SIMPLE ASSAULT", "ASSAULT", "*");
        crimeCodeHierarchy.add("ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT", "ASSAULT", "*");
        crimeCodeHierarchy.add("INTIMATE PARTNER - SIMPLE ASSAULT", "ASSAULT", "*");
        crimeCodeHierarchy.add("INTIMATE PARTNER - AGGRAVATED ASSAULT", "ASSAULT", "*");
        crimeCodeHierarchy.add("CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT", "ASSAULT", "*");
        crimeCodeHierarchy.add("BATTERY WITH SEXUAL CONTACT", "SEXUAL ASSAULT", "*");

        // Sexual crimes
        crimeCodeHierarchy.add("RAPE, FORCIBLE", "SEXUAL ASSAULT", "*");
        crimeCodeHierarchy.add("ORAL COPULATION", "SEXUAL ASSAULT", "*");
        crimeCodeHierarchy.add("SEXUAL PENETRATION W/FOREIGN OBJECT", "SEXUAL ASSAULT", "*");
        crimeCodeHierarchy.add("SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ", "SEXUAL ASSAULT", "*");
        crimeCodeHierarchy.add("SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH", "SEXUAL ASSAULT", "*");
        crimeCodeHierarchy.add("LEWD CONDUCT", "SEXUAL ASSAULT", "*");

        // Crimes against children
        crimeCodeHierarchy.add("CRM AGNST CHLD (13 OR UNDER) (14-15 & SUSP 10 YRS OLDER)", "CHILD-RELATED CRIME", "*");
        crimeCodeHierarchy.add("CHILD ANNOYING (17YRS & UNDER)", "CHILD-RELATED CRIME", "*");

        // Miscellaneous crimes
        crimeCodeHierarchy.add("CRIMINAL THREATS - NO WEAPON DISPLAYED", "THREATS", "*");
        crimeCodeHierarchy.add("LETTERS, LEWD  -  TELEPHONE CALLS, LEWD", "HARASSMENT", "*");
        crimeCodeHierarchy.add("UNAUTHORIZED COMPUTER ACCESS", "CYBERCRIME", "*");
        crimeCodeHierarchy.add("OTHER MISCELLANEOUS CRIME", "MISCELLANEOUS", "*");
        crimeCodeHierarchy.add("DOCUMENT FORGERY / STOLEN FELONY", "FRAUD", "*");
        crimeCodeHierarchy.add("EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)", "FRAUD", "*");
        crimeCodeHierarchy.add("EXTORTION", "THREATS", "*");
        crimeCodeHierarchy.add("VIOLATION OF RESTRAINING ORDER", "HARASSMENT", "*");
        crimeCodeHierarchy.add("SEX OFFENDER REGISTRANT OUT OF COMPLIANCE", "SEXUAL OFFENSE", "*");
        crimeCodeHierarchy.add("VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)", "PROPERTY DAMAGE", "*");
        crimeCodeHierarchy.add("VANDALISM - MISDEAMEANOR ($399 OR UNDER)", "PROPERTY DAMAGE", "*");
        crimeCodeHierarchy.add("BUNCO, GRAND THEFT", "FRAUD", "*");

        return crimeCodeHierarchy;
    }

    private static void latHierarchy(int numIntervals) {
        String[] latValues = getLATValues();
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (String value : latValues) {
            if (value != null && !value.isEmpty()) {
                double numValue = Double.parseDouble(value);
                if (numValue < min) min = numValue;
                if (numValue > max) max = numValue;
            }
        }

        // Interval size
        double intervalSize = (max - min) / numIntervals;

        latBuilder = HierarchyBuilderIntervalBased.create(DataType.DECIMAL);

        double currentStart = min;
        for (int i = 0; i < numIntervals; i++) {
            double currentEnd = currentStart + intervalSize;

            if (i == numIntervals - 1) {
                currentEnd = max + 0.000001; // Add a small value to include max
            }

            latBuilder.addInterval(currentStart, currentEnd);

            currentStart = currentEnd;
        }

        latBuilder.getLevel(0).addGroup(2); // Example: Grouping for first level
        latBuilder.getLevel(1).addGroup(3); // Example: Grouping for second level

        System.out.println("--------------------------");
        System.out.println("LAT HIERARCHY");
        System.out.println("--------------------------");
        System.out.println("");
        System.out.println("SPECIFICATION");

        for (Interval<Double> interval : latBuilder.getIntervals()) {
            System.out.println(interval);
        }

        for (Level<Double> level : latBuilder.getLevels()) {
            System.out.println(level);
        }

        // Prepare and print resulting levels
        System.out.println("Resulting levels: " + Arrays.toString(latBuilder.prepare(latValues)));
        System.out.println("");
    }
    /**
    private static void latHierarchy() {

        // Create the builder
      latBuilder = HierarchyBuilderIntervalBased.create(DataType.DECIMAL);

        // Define base intervals
        latBuilder.addInterval(0.0d, 30.0d, "<30");
        latBuilder.addInterval(30.0d, 30.5d, "30-30.5");
        latBuilder.addInterval(30.5d, 31.0d, "30.5-31");
        latBuilder.addInterval(31.0d, 31.5d, "31-31.5");
        latBuilder.addInterval(31.5d, 32.0d, "31.5-32");
        latBuilder.addInterval(32.0d, 32.5d, "32-32.5");
        latBuilder.addInterval(32.5d, 33.0d, "32.5-33");
        latBuilder.addInterval(33.0d, 33.5d, "33-33.5");
        latBuilder.addInterval(33.5d, 34.0d, "33.5-34");
        latBuilder.addInterval(34.0d, 34.5d, "34-34.5");
        latBuilder.addInterval(34.5d, 35.0d, "34.5-35");
        latBuilder.addInterval(35.0d, 35.5d, "35-35.5");
        latBuilder.addInterval(35.5d, 100.0, ">35");


        // Define grouping fanouts
        latBuilder.getLevel(0).addGroup(2);
        latBuilder.getLevel(1).addGroup(3);


        System.out.println("--------------------------");
        System.out.println("LAT HIERARCHY");
        System.out.println("--------------------------");
        System.out.println("");
        System.out.println("SPECIFICATION");

        // Print specification
        for (Interval<Double> interval : latBuilder.getIntervals()){
            System.out.println(interval);
        }

        // Print specification
        for (Level<Double> level : latBuilder.getLevels()) {
            System.out.println(level);
        }

        // Print info about resulting levels
        System.out.println("Resulting levels: "+Arrays.toString(latBuilder.prepare(getLATValues())));

        System.out.println("");
        System.out.println("RESULT");

        // Print resulting hierarchy
        //printArray(builder.build().getHierarchy());
        System.out.println("");

    }
    **/

    private static String[] getLATValues() {
        DataHandle handle = data.getHandle();
        int columnIndex = handle.getColumnIndexOf("LAT"); // Find the index of the LAT column
        String[] latValues = new String[handle.getNumRows()];

        for (int i = 0; i < handle.getNumRows(); i++) {
            latValues[i] = handle.getValue(i, columnIndex);
        }
        return latValues;
    }


    private static void lonHierarchy() {

        // Create the builder
        lonBuilder = HierarchyBuilderIntervalBased.create(DataType.DECIMAL);

        // Define base intervals
        lonBuilder.addInterval(-200.0d, -119.0d, "<-119");
        lonBuilder.addInterval(-119.0d, -118.9d, "-119 to -118.9");
        lonBuilder.addInterval(-118.9d, -118.8d, "-118.9 to -118.8");
        lonBuilder.addInterval(-118.8d, -118.7d, "-118.8 to -118.7");
        lonBuilder.addInterval(-118.7d, -118.6d, "-118.7 to -118.6");
        lonBuilder.addInterval(-118.6d, -118.5d, "-118.6 to -118.5");
        lonBuilder.addInterval(-118.5d, -118.4d, "-118.5 to -118.4");
        lonBuilder.addInterval(-118.4d, -118.3d, "-118.4 to -118.3");
        lonBuilder.addInterval(-118.3d, -118.2d, "-118.3 to -118.2");
        lonBuilder.addInterval(-118.2d, -118.1d, "-118.2 to -118.1");
        lonBuilder.addInterval(-118.1d, -118.0d, "-118.1 to -118.0");
        lonBuilder.addInterval(-118.0d, 200.0d, ">-118");


        // Define grouping fanouts
        lonBuilder.getLevel(0).addGroup(2);
        lonBuilder.getLevel(1).addGroup(3);


        System.out.println("--------------------------");
        System.out.println("LON HIERARCHY");
        System.out.println("--------------------------");
        System.out.println("");
        System.out.println("SPECIFICATION");

        // Print specification
        for (Interval<Double> interval : lonBuilder.getIntervals()){
            System.out.println(interval);
        }

        // Print specification
        for (Level<Double> level : lonBuilder.getLevels()) {
            System.out.println(level);
        }

        // Print info about resulting levels
        System.out.println("Resulting levels: "+Arrays.toString(lonBuilder.prepare(getLONValues())));

        System.out.println("");
        System.out.println("RESULT");

        // Print resulting hierarchy
        //printArray(lonBuilder.build().getHierarchy());
        System.out.println("");

    }

    private static String[] getLONValues() {
        DataHandle handle = data.getHandle();
        int columnIndex = handle.getColumnIndexOf("LON"); // Find the index of the LON column
        String[] lonValues = new String[handle.getNumRows()];

        for (int i = 0; i < handle.getNumRows(); i++) {
            lonValues[i] = handle.getValue(i, columnIndex);
        }
        return lonValues;
    }


    private static void printArray(String[][] array) {
        for (String[] row : array) {
            for (String cell : row) {
                System.out.print(cell + "\t");
            }
            System.out.println();
        }
    }


}