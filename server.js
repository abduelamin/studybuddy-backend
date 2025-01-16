import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { readFile } from "fs/promises";

dotenv.config();
const app = express();
app.use(express.json());
const openAIApiKey = process.env.OPENAI_API_KEY;
const openai = new OpenAI({
  apiKey: openAIApiKey,
});
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_API_KEY
);

const textsplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});

// This function controls fetching the knowledge base, splitting it, creating chunks and storing chunks in the database.
const fetchbase = async () => {
  try {
    // this is how you fetch a text file in node, its idfferent than when using vite as vite allows direct imports.
    const base = await readFile(new URL("./base.txt", import.meta.url), "utf8");

    const rawtexts = await textsplitter.createDocuments([base]);
    const texts = await textsplitter.splitDocuments(rawtexts);
    const splitText = texts.map((doc) => doc.pageContent);

    const docEmbeddings = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: splitText,
    });

    const dataTostore = docEmbeddings.data.map((embedding, index) => ({
      content: splitText[index],
      embedding: embedding.embedding,
      metadata: texts[index].metadata,
    }));

    const { data, error } = await supabase
      .from("documents")
      .insert(dataTostore);

    if (error) {
      console.error("Error inserting data into Supabase:", error);
    } else {
      console.log("Data successfully inserted into Supabase:", data);
    }
  } catch (error) {
    console.error("Error fetching/splitting:", error);
  }
};

// fetchbase(); // Only call this function if you updated your txt file.

app.listen(80, (req, res) => {
  console.log("Server is running on PORT 80");
});
