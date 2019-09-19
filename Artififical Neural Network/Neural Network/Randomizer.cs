using System;

public static class Randomizer
{
    private static readonly Random random = new Random(DateTime.Now.Second + DateTime.Now.Day);

    /// <summary>
    /// Get a random bool betwen miniumum and maximum.
    /// </summary>
    /// <param name="minimum"></param>
    /// <param name="maximum"></param>
    /// <returns></returns>
    public static double GetRandomNumber(double minimum, double maximum)
    {
        return random.NextDouble() * (maximum - minimum) + minimum;
    }
}

