import "./globals.css";

export const metadata = {
  title: "Multi-Agent AI Analyst",
  description: "LangGraph supervisor + specialist agents + critic, live trace",
};

export default function RootLayout({ children }) {
  return (
    <html lang="uz">
      <body>{children}</body>
    </html>
  );
}