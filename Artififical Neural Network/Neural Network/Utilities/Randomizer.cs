using System;

public static class Randomizer
{
    public static readonly Random random = new Random((int)DateTime.Now.Ticks);

    /// <summary>
    /// Get a random double betwen miniumum and maximum.
    /// </summary>
    /// <param name="minimum"></param>
    /// <param name="maximum"></param>
    /// <returns></returns>
    public static double Range(double minimum, double maximum)
    {
        return random.NextDouble() * (maximum - minimum) + minimum;
    }
}

