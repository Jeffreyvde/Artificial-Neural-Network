using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using NeuralNetworks;


namespace Image
{
    public class ImageTrainingSet
    {
        public static TrainingData Process(string path, byte correct = 0)
        {
            RemoveLineEndings(path);
            // The byte[] to save the data in
            string test = File.ReadAllText(path);
            double[] image = new double[test.Length];

            for (int i = 0; i < test.Length; i++)
            {
                image[i] = test[i] == '0' ? 0 : 1;
            }

            TrainingData trainingData = new TrainingData(image, correct);
            return trainingData;
        }


        private static void RemoveLineEndings(string path)
        {
            string readText = File.ReadAllText(path);
            string replacement = Regex.Replace(readText, @"\t|\n|\r", "");
            File.WriteAllText(path, replacement);
        }
    }
}
