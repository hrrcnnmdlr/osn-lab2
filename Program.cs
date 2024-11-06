using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace WaterQualityPrediction
{
    // Class to represent the input data for the water quality prediction model
    public class WaterData
    {
        [LoadColumn(0)]
        public float pH { get; set; } // The pH level of the water

        [LoadColumn(1)]
        public float Hardness { get; set; } // The hardness of the water

        [LoadColumn(2)]
        public float Solids { get; set; } // Total dissolved solids in the water

        [LoadColumn(3)]
        public float Chloramines { get; set; } // Chloramines level in the water

        [LoadColumn(4)]
        public float Sulfate { get; set; } // Sulfate concentration in the water

        [LoadColumn(5)]
        public float Conductivity { get; set; } // Conductivity level of the water

        [LoadColumn(6)]
        public float Organic_carbon { get; set; } // Organic carbon concentration

        [LoadColumn(7)]
        public float Trihalomethanes { get; set; } // Trihalomethanes concentration

        [LoadColumn(8)]
        public float Turbidity { get; set; } // Turbidity level of the water

        [LoadColumn(9)]
        public bool Potability { get; set; } // Indicates whether the water is potable (true) or not (false)
    }

    // Class to represent the prediction results
    public class WaterQualityPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Potability { get; set; } // Predicted potability of the water
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Create a new ML context for the application
            var context = new MLContext();

            // Specify the path to the CSV data file
            string dataPath = "C:\\Users\\stere\\source\\repos\\Lab2\\water_potability.csv"; // Ensure the correct path to the CSV file
            var dataView = context.Data.LoadFromTextFile<WaterData>(dataPath, separatorChar: ',', hasHeader: true);


            // Split the data into training and testing sets (80% training, 20% testing)
            var split = context.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            // Build a pipeline for training the model with normalization
            var pipeline = context.Transforms.Concatenate("Features", nameof(WaterData.pH), nameof(WaterData.Hardness),
                                                           nameof(WaterData.Solids), nameof(WaterData.Chloramines),
                                                           nameof(WaterData.Sulfate), nameof(WaterData.Conductivity),
                                                           nameof(WaterData.Organic_carbon), nameof(WaterData.Trihalomethanes),
                                                           nameof(WaterData.Turbidity))
                .Append(context.Transforms.NormalizeMinMax("Features"))
                .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: nameof(WaterData.Potability),
                                                                         numberOfLeaves: 20,
                                                                         minimumExampleCountPerLeaf: 10,
                                                                         learningRate: 0.2));

            // Train the model using the training data
            var model = pipeline.Fit(trainData);

            // Evaluate the model using the test data
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(WaterData.Potability));

            // Display evaluation metrics
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}"); // Print the accuracy of the model
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}"); // Print the area under the ROC curve
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}"); // Print the F1 score

            // Check if the model performance is acceptable based on accuracy
            if (metrics.Accuracy >= 0.7) // Setting the threshold at 70%
            {
                Console.WriteLine("The model performs well and can be used for prediction on this type of data.");
            }
            else
            {
                Console.WriteLine("The model may not be accurate enough for reliable predictions. Consider improving the model.");
            }

            // Create a sample data instance for prediction
            var sampleData = new WaterData()
            {
                pH = 7.0f,
                Hardness = 200.0f,
                Solids = 15000.0f,
                Chloramines = 8.0f,
                Sulfate = 350.0f,
                Conductivity = 400.0f,
                Organic_carbon = 10.0f,
                Trihalomethanes = 3.0f,
                Turbidity = 2.0f
            };

            // Create a prediction engine for making predictions
            var predictionEngine = context.Model.CreatePredictionEngine<WaterData, WaterQualityPrediction>(model);
            var prediction = predictionEngine.Predict(sampleData); // Make a prediction on the sample data

            // Output the prediction result
            Console.WriteLine($"Predicted Potability: {prediction.Potability}");

            // Wait for user input before exiting
            Console.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }
    }
}
