/*
 * Copyright (c) 2018, Jeffrey van den Elshout
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

/// <summary>
/// This class can be used to save to a specific file
/// </summary>
public class IOFile
{
    public string Path { get; private set; }

    public string Location { get; private set; }

    public string FileName { get; private set; }
    public string Extension { get; private set; }

    private readonly BinaryFormatter binaryFormatter;

    /// <summary>
    /// Create an IOFile.
    /// </summary>
    /// <param name="location">Location where you want the File to save</param>
    /// <param name="fileName">Name of this File</param>
    /// <param name="extension">The extension of your File </param>
    public IOFile(string location, string fileName, string extension)
    {
        Location = location ?? throw new ArgumentNullException(nameof(location));
        FileName = fileName ?? throw new ArgumentNullException(nameof(fileName));
        Extension = extension ?? throw new ArgumentNullException(nameof(extension));

        binaryFormatter = binaryFormatter = new BinaryFormatter();

        if (extension[0] == '.')
            extension = extension.Remove(0);

        Path = System.IO.Path.Combine(location, fileName) + "." + extension;

    }


    /// <summary>
    /// Check if a file exists
    /// </summary>
    /// <returns>True if the file from this IOFIle exists false if it doesnt</returns>
    public bool Exists()
    {
        return File.Exists(Path);
    }

    /// <summary>
    /// Standard version of saving data.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="data"></param>
    public void Save(object data)
    {
        using (Stream stream = File.Open(Path, FileMode.OpenOrCreate))
        {
            binaryFormatter.Serialize(stream, data);
        }
    }

    /// <summary>
    /// Load a class from the specified IO path
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns>The data on the File if it exists</returns>
    public T Load<T>()
    {
        using (Stream stream = File.Open(Path, FileMode.Open))
        {
            return (T)binaryFormatter.Deserialize(stream);
        }
    }

    /// <summary>
    /// Load a file 
    /// </summary>
    /// <returns>The loaded object</returns>
    public object Load()
    {
        byte[] data = File.ReadAllBytes(Path);
        using (MemoryStream ms = new MemoryStream(data))
        {
            return binaryFormatter.Deserialize(ms);
        }
    }

    /// <summary>
    /// Delete the IOFile
    /// </summary>
    public void Delete()
    {
        File.Delete(Path);
    }
}

