using System;
using System.IO;
using NeuralNetwork;


namespace MNIST
{
    public class MnistDataSet
    {
        private readonly int numRows, numCols;
        public readonly TrainingData[] trainingData;

        public MnistDataSet(string labels, string images)
        {
            FileStream fsLabels = new FileStream(labels, FileMode.Open);
            FileStream fsImages = new FileStream(images, FileMode.Open);

            BinaryReader brLabels = new BinaryReader(fsLabels);
            BinaryReader brImages = new BinaryReader(fsImages);

            Discard(brImages);
            trainingData = new TrainingData[brImages.ReadBigInt32()];
            numRows = brImages.ReadBigInt32();
            numCols = brImages.ReadBigInt32();

            Discard(brLabels, 2);

            // each test image
            for (int i = 0; i < trainingData.Length; ++i)
            {
                trainingData[i] = LoadTrainingData(brImages, brLabels);
            }

            //Close all streams
            fsImages.Close();
            brImages.Close();
            fsLabels.Close();
            brLabels.Close();
        }

        private void Discard(BinaryReader binaryReader, int times = 1)
        {
            for (int i = 0; i < times; i++)
            {
                binaryReader.ReadInt32();
            }
        }

        /// <summary>
        /// Load new training data
        /// </summary>
        /// <param name="image"></param>
        /// <param name="label"></param>
        /// <returns></returns>
        private TrainingData LoadTrainingData(BinaryReader image, BinaryReader label)
        {
            double[] data = new double[numRows * numCols];
            for (int x = 0; x < numRows; ++x)
            {
                for (int y = 0; y < numCols; ++y)
                {
                    data[x + (y * numCols)] = image.ReadByte() / 255f;
                }
            }
            return new TrainingData(data, label.ReadByte());
        }
    }
}
