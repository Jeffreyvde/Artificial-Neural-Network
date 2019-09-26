using System;
using System.IO;
using NeuralNetworks;


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

        public static void Load(string labels, string images)
        {
            try
            {
                Console.WriteLine("\nBegin\n");
                FileStream ifsLabels = new FileStream(labels, FileMode.Open);
                FileStream ifsImages = new FileStream(images, FileMode.Open);

                BinaryReader brLabels = new BinaryReader(ifsLabels);
                BinaryReader brImages = new BinaryReader(ifsImages);

                int magicNumberImages = brImages.ReadBigInt32(); // discard
                int numImages = brImages.ReadBigInt32();
                int numRows = brImages.ReadBigInt32();
                int numCols = brImages.ReadBigInt32();

                int magicNumberLabels = brLabels.ReadBigInt32();
                int numLabels = brLabels.ReadBigInt32();

                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                // each test image
                for (int di = 0; di < numImages; di++)
                {
                    for (int i = 0; i < numRows; i++)
                    {
                        for (int j = 0; j < numCols; j++)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();

                    DigitImage dImage = new DigitImage(pixels, lbl);
                    Console.WriteLine(dImage.ToString());
                    Console.ReadLine();
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                Console.WriteLine("\nEnd\n");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
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
            byte[] data = new byte[numRows * numCols];
            for (int x = 0; x < numRows; ++x)
            {
                for (int y = 0; y < numCols; ++y)
                {
                    data[x + (y * numCols)] = image.ReadByte();
                }
            }
            return new TrainingData(data, label.ReadByte());
        }
    }

    public class DigitImage
    {
        public byte[][] pixels;
        public byte label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

            this.label = label;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i][j] == 0)
                        s += " "; // white
                    else if (this.pixels[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } // ToString

    }
}
