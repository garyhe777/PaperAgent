import fs from "node:fs/promises";
import path from "node:path";

const renderConfigPath = process.argv[2];

if (!renderConfigPath) {
  throw new Error("Missing render_config.json path.");
}

const { FileBlob, Presentation, PresentationFile } = await import("@oai/artifact-tool");

const renderConfig = JSON.parse(await fs.readFile(renderConfigPath, "utf-8"));
const deckContent = JSON.parse(await fs.readFile(renderConfig.content_path, "utf-8"));

const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

for (const slideSpec of deckContent.slides) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  const titleShape = slide.shapes.add({
    geometry: "rect",
    position: { left: 60, top: 40, width: 1160, height: 80 },
    fill: "#F8FAFC",
    line: { width: 0, fill: "#F8FAFC" },
  });
  titleShape.text = slideSpec.title;
  titleShape.text.fontSize = slideSpec.type === "title" ? 34 : 28;
  titleShape.text.bold = true;

  const bodyShape = slide.shapes.add({
    geometry: "rect",
    position: { left: 80, top: 150, width: 1080, height: 420 },
    fill: "#FFFFFF",
    line: { width: 1, fill: "#CBD5E1" },
  });
  bodyShape.text = slideSpec.bullets.join("\n");
  bodyShape.text.fontSize = 22;
  bodyShape.text.insets = { left: 20, right: 20, top: 20, bottom: 20 };

  if (slideSpec.notes) {
    slide.speakerNotes.setText(slideSpec.notes);
  }
}

const outputDir = path.dirname(renderConfig.output_path);
await fs.mkdir(outputDir, { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(renderConfig.output_path);
