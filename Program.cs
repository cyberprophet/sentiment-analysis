namespace SentimentAnalysis;

class Program
{
    static void Main()
    {
        
    }
    string _dataPath = Path.Combine(
        Directory.GetParent(
        Directory.GetParent(        Environment.CurrentDirectory)!.FullName)!.FullName
        , "Data", "yelp_labelled.text");
}