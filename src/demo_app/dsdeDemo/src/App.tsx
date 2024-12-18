import { useState } from "react";
import { TextField, Button, Typography, LinearProgress } from "@mui/material";
import "./App.css";

interface Result {
  predicted_subject_area: [string];
  title: [string];
  abstract: [string];
  keywords: [string];
}

function App() {
  const [title, setTitle] = useState("");
  const [abstract, setAbstract] = useState("");
  const [keywords, setKeywords] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    const data = {
      title: [title],
      abstract: [abstract],
      keywords: [keywords],
    };

    console.log(data);

    try {
      const response = await fetch("http://127.0.0.1:5000/pred", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      const res = await response.json();
      setResult(res);
      console.log(res);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen font-serif">
      <div className="text-4xl text-sky-100 font-serif">
        Predict the Category of a Research Paper
      </div>
      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-4 w-full max-w-md bg-red-100 p-6 rounded-lg shadow-lg mt-12"
      >
        <TextField
          label="Title"
          variant="outlined"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full"
        />
        <TextField
          label="Abstract"
          variant="outlined"
          multiline
          rows={4}
          value={abstract}
          onChange={(e) => setAbstract(e.target.value)}
          className="w-full"
        />
        <TextField
          label="Keywords"
          variant="outlined"
          value={keywords}
          onChange={(e) => setKeywords(e.target.value)}
          className="w-full"
        />
        <Button
          type="submit"
          variant="contained"
          color="success"
          className="w-full bg-black"
        >
          Predict
        </Button>

        {/* Result Box */}
        <div className="flex-1 bg-gray-50 p-6 rounded-lg shadow-lg overflow-y-auto">
          <Typography variant="h6" className="mb-4 text-gray-700">
            Result
          </Typography>
          {loading ? (
            <div className="p-8"><LinearProgress /></div>
          ) : (
            <pre className="text-gray-600 pt-2">
              {!result ? (
                "The result will appear here..."
              ) : (
                <div>
                  <p className="text-ellipsis whitespace-normal">
                    <strong>Predicted Subject Area:</strong>{" "}
                    {result?.predicted_subject_area}
                  </p>
                </div>
              )}
            </pre>
          )}
        </div>

        {/* Clear data button */}
        <Button
          onClick={() => {
            setTitle("");
            setAbstract("");
            setKeywords("");
            setResult(null);
          }}
          variant="contained"
          color="error"
          className="w-full"
        >Clear</Button>
      </form>
    </div>
  );
}

export default App;
