import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { readFile } from "fs/promises";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";

dotenv.config();
const app = express();
app.use(express.json());
app.use(
  cors({
    origin: "https://studybuddy-ai.netlify.app",
    credentials: true,
    methods: ["GET", "POST"],
  })
);

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

// standalone question logic
const embeddings = new OpenAIEmbeddings({ openAIApiKey: openAIApiKey });

const vectorStore = new SupabaseVectorStore(
  embeddings,

  {
    client: supabase,
    tableName: "documents",
    queryName: "match_documents",
  }
);

const model = new ChatOpenAI({ openAIApiKey: openAIApiKey });

const retriever = vectorStore.asRetriever();

const questionTemplate = `I want you to convert the user question to a standalone question. If applicable use conversation history for extra bit of help. 
  question: {question}
  conversation history: {chatHistory}
  standalone question: `;

const questionPrompt = PromptTemplate.fromTemplate(questionTemplate);

const questionChain = questionPrompt.pipe(model).pipe(new StringOutputParser());

// below function formats the comparison vectors and places them into one big string. From there we can compare the vectorstore chunks to the standalonequestion vector.
const formatDocuments = (documents) =>
  documents.map((doc) => doc.pageContent).join("\n");
const retrieverChain = RunnableSequence.from([
  (prevResult) => prevResult.standaloneQuestion,
  retriever,
  formatDocuments,
]);

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the user to email abdullahelamin.dev@gmail.com for assistance. Don't try to make up an answer. Always speak as if you were chatting to a friend.
  context: {context}
  conversationHistory: {chatHistory}
  question: {question}
  answer: `;

const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

const answerChain = answerPrompt.pipe(model).pipe(new StringOutputParser());

const mainChain = RunnableSequence.from([
  {
    standaloneQuestion: questionChain,
    input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain, // Prev result is the previous output object. We then access the standalonequestion via dot notation in the retrieverChain
    chatHistory: ({ input }) => input.chatHistory,
    question: ({ input }) => input.question,
  },
  answerChain,
]);

app.post("/api/chat", async (req, res) => {
  const { question, chatHistory } = req.body;
  try {
    const response = await mainChain.invoke({
      question,
      chatHistory,
    });
    res.status(200).json({ response });
  } catch (error) {
    console.error(err);
    res.status(500).json({ error: "failed to process the question" });
  }
});

app.listen(80, (req, res) => {
  console.log("Server is running on PORT 80");
});
