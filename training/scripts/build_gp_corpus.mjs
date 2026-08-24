import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

import * as alphaSkia from '@coderline/alphaskia';
import * as alphaTab from '@coderline/alphatab';
import sharp from 'sharp';

const SUPPORTED_EXTENSIONS = new Set(['.gp', '.gp3', '.gp4', '.gp5', '.gpx']);

function parseArgs(argv) {
  const options = {
    source: 'D:\\document\\谱子',
    output: path.resolve('training/data/private-gp'),
    limit: Number.POSITIVE_INFINITY,
    width: 1600,
    allTracks: false,
    profile: 'tab',
    match: null,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--source') options.source = argv[++index];
    else if (arg === '--output') options.output = path.resolve(argv[++index]);
    else if (arg === '--limit') options.limit = Number(argv[++index]);
    else if (arg === '--width') options.width = Number(argv[++index]);
    else if (arg === '--all-tracks') options.allTracks = true;
    else if (arg === '--profile') options.profile = argv[++index].toLowerCase();
    else if (arg === '--match') options.match = argv[++index];
    else if (arg === '--help') {
      console.log('node training/scripts/build_gp_corpus.mjs [--source DIR] [--output DIR] [--limit N] [--width PX] [--all-tracks] [--profile tab|scoretab] [--match TEXT]');
      process.exit(0);
    } else {
      throw new Error(`未知参数: ${arg}`);
    }
  }

  if (
    options.limit !== Number.POSITIVE_INFINITY
    && (options.limit <= 0 || !Number.isInteger(options.limit))
  ) {
    throw new Error('--limit 必须是正整数');
  }
  if (!Number.isInteger(options.width) || options.width < 600 || options.width > 5000) {
    throw new Error('--width 必须是 600 到 5000 的整数');
  }
  if (!['tab', 'scoretab'].includes(options.profile)) throw new Error('--profile 只支持 tab 或 scoretab');
  return options;
}

function walkFiles(root) {
  const files = [];
  const pending = [root];
  while (pending.length) {
    const current = pending.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) pending.push(fullPath);
      else if (SUPPORTED_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) files.push(fullPath);
    }
  }
  return files.sort((left, right) => left.localeCompare(right, 'zh-CN'));
}

function exactArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

function initializeSkia() {
  const bravuraPath = path.resolve('node_modules/@coderline/alphatab/dist/font/Bravura.otf');
  alphaTab.Environment.enableAlphaSkia(exactArrayBuffer(fs.readFileSync(bravuraPath)), alphaSkia);
  const fontPaths = [
    'C:\\Windows\\Fonts\\arial.ttf',
    'C:\\Windows\\Fonts\\arialbd.ttf',
    'C:\\Windows\\Fonts\\ariali.ttf',
    'C:\\Windows\\Fonts\\arialbi.ttf',
  ];
  return fontPaths.map((fontPath) =>
    alphaTab.Environment.registerAlphaSkiaCustomFont(new Uint8Array(fs.readFileSync(fontPath))),
  );
}

function applyTextFonts(settings, fontInfo) {
  const regular = fontInfo[0].families;
  const bold = fontInfo[1].families;
  const italic = fontInfo[2].families;
  const resources = settings.display.resources;
  resources.copyrightFont.families = regular;
  resources.titleFont.families = bold;
  resources.subTitleFont.families = regular;
  resources.wordsFont.families = regular;
  resources.effectFont.families = regular;
  resources.fretboardNumberFont.families = regular;
  resources.tablatureFont.families = regular;
  resources.graceFont.families = regular;
  resources.barNumberFont.families = regular;
  resources.fingeringFont.families = italic;
  resources.markerFont.families = bold;
}

function serializeBounds(bounds) {
  return {
    x: Math.round(bounds.x * 1000) / 1000,
    y: Math.round(bounds.y * 1000) / 1000,
    w: Math.round(bounds.w * 1000) / 1000,
    h: Math.round(bounds.h * 1000) / 1000,
  };
}

function serializeNote(noteBounds) {
  const note = noteBounds.note;
  return {
    note_id: note.id,
    note_index: note.index,
    box: serializeBounds(noteBounds.noteHeadBounds),
    fret: note.fret,
    string: note.string,
    octave: note.octave,
    tone: note.tone,
    is_stringed: note.isStringed,
    is_dead: note.isDead,
    is_ghost: note.isGhost,
    is_tie_origin: note.isTieOrigin,
    is_tie_destination: note.isTieDestination,
    is_hammer_pull_origin: note.isHammerPullOrigin,
    is_hammer_pull_destination: note.isHammerPullDestination,
    slide_in_type: note.slideInType,
    slide_out_type: note.slideOutType,
    has_bend: note.hasBend,
    bend_type: note.bendType,
    vibrato: note.vibrato,
    harmonic_type: note.harmonicType,
    is_palm_mute: note.isPalmMute,
    is_let_ring: note.isLetRing,
  };
}

function serializeLookup(lookup) {
  const systems = [];
  let beatCount = 0;
  let noteCount = 0;
  for (const system of lookup?.staffSystems ?? []) {
    const masterBars = [];
    for (const masterBar of system.bars) {
      const bars = [];
      for (const barBounds of masterBar.bars) {
        const beats = [];
        for (const beatBounds of barBounds.beats) {
          const beat = beatBounds.beat;
          const notes = (beatBounds.notes ?? []).map(serializeNote);
          noteCount += notes.length;
          beatCount += 1;
          beats.push({
            beat_id: beat.id,
            beat_index: beat.index,
            voice_index: beat.voice?.index ?? null,
            duration: beat.duration,
            dots: beat.dots,
            display_start_ticks: beat.displayStart,
            display_duration_ticks: beat.displayDuration,
            playback_start_ticks: beat.playbackStart,
            playback_duration_ticks: beat.playbackDuration,
            absolute_display_start_ticks: beat.absoluteDisplayStart,
            absolute_playback_start_ticks: beat.absolutePlaybackStart,
            is_rest: beat.isRest,
            is_full_bar_rest: beat.isFullBarRest,
            is_empty: beat.isEmpty,
            is_palm_mute: beat.isPalmMute,
            is_let_ring: beat.isLetRing,
            on_notes_x: Math.round(beatBounds.onNotesX * 1000) / 1000,
            visual_box: serializeBounds(beatBounds.visualBounds),
            real_box: serializeBounds(beatBounds.realBounds),
            notes,
          });
        }
        bars.push({
          staff_index: barBounds.bar.staff.index,
          bar_index: barBounds.bar.index,
          visual_box: serializeBounds(barBounds.visualBounds),
          real_box: serializeBounds(barBounds.realBounds),
          beats,
        });
      }
      masterBars.push({
        master_bar_index: masterBar.index,
        time_signature_numerator: masterBar.bars[0]?.bar?.masterBar?.timeSignatureNumerator ?? null,
        time_signature_denominator: masterBar.bars[0]?.bar?.masterBar?.timeSignatureDenominator ?? null,
        is_first_of_line: masterBar.isFirstOfLine,
        visual_box: serializeBounds(masterBar.visualBounds),
        real_box: serializeBounds(masterBar.realBounds),
        line_box: serializeBounds(masterBar.lineAlignedBounds),
        bars,
      });
    }
    systems.push({
      system_index: system.index,
      visual_box: serializeBounds(system.visualBounds),
      real_box: serializeBounds(system.realBounds),
      master_bars: masterBars,
    });
  }
  return { systems, beat_count: beatCount, note_box_count: noteCount };
}

async function renderTrack(score, trackIndex, options, fontInfo) {
  const settings = new alphaTab.Settings();
  settings.core.engine = 'skia';
  settings.core.includeNoteBounds = true;
  settings.core.enableLazyLoading = false;
  settings.display.staveProfile = options.profile === 'tab' ? alphaTab.StaveProfile.Tab : alphaTab.StaveProfile.ScoreTab;
  applyTextFonts(settings, fontInfo);

  const renderer = new alphaTab.rendering.ScoreRenderer(settings);
  renderer.width = options.width;
  const partialIds = [];
  const renderedChunks = [];
  let finalSize = null;
  let renderError = null;
  let renderedPartials = 0;

  renderer.partialLayoutFinished.on((result) => partialIds.push(result.id));
  renderer.error.on((error) => {
    renderError = error;
  });
  renderer.renderFinished.on((result) => {
    finalSize = { width: result.totalWidth, height: result.totalHeight };
    for (const id of partialIds) renderer.renderResult(id);
  });
  renderer.partialRenderFinished.on((result) => {
    renderedChunks.push({
      png: Buffer.from(result.renderResult.toPng()),
      x: Math.round(result.x),
      y: Math.round(result.y),
    });
    renderedPartials += 1;
    result.renderResult[Symbol.dispose]();
  });

  renderer.renderScore(score, [trackIndex]);
  if (renderError) throw renderError;
  if (!finalSize) throw new Error('alphaTab 未返回最终渲染尺寸');
  if (renderedPartials === 0) throw new Error('alphaTab 未返回任何图像分片');
  const png = await sharp({
    create: {
      width: Math.ceil(finalSize.width),
      height: Math.ceil(finalSize.height),
      channels: 4,
      background: { r: 255, g: 255, b: 255, alpha: 1 },
    },
  })
    .composite(renderedChunks.map((chunk) => ({ input: chunk.png, left: chunk.x, top: chunk.y })))
    .png()
    .toBuffer();
  const labels = serializeLookup(renderer.boundsLookup);
  renderer.destroy();
  return { png, finalSize, labels, renderedPartials };
}

function trackSummary(track) {
  return {
    index: track.index,
    name: track.name,
    short_name: track.shortName,
    program: track.playbackInfo?.program ?? null,
    staves: track.staves.map((staff) => ({
      index: staff.index,
      bar_count: staff.bars.length,
      is_stringed: staff.isStringed,
      tuning: staff.tuning,
      tuning_name: staff.tuningName,
      show_tablature: staff.showTablature,
      show_standard_notation: staff.showStandardNotation,
    })),
  };
}

function countStringedNotes(track) {
  let count = 0;
  for (const staff of track.staves) {
    for (const bar of staff.bars) {
      for (const voice of bar.voices) {
        for (const beat of voice.beats) {
          for (const note of beat.notes) {
            if (note.isStringed && note.string > 0) count += 1;
          }
        }
      }
    }
  }
  return count;
}

function preferredStringedTrackIndexes(score) {
  const instrumentPattern = /(guitar|gtr|bass|吉他|贝斯|ギター|ベース|distortion|overdriven|clean|acoustic|lead|rhythm)/i;
  return score.tracks
    .map((track) => {
      const stringedNotes = countStringedNotes(track);
      const scoreValue =
        (instrumentPattern.test(`${track.name} ${track.shortName}`) ? 100_000 : 0)
        + (track.staves.some((staff) => staff.showTablature) ? 10_000 : 0)
        + stringedNotes;
      return { index: track.index, stringedNotes, scoreValue };
    })
    .filter((item) => item.stringedNotes > 0)
    .sort((left, right) => right.scoreValue - left.scoreValue || left.index - right.index)
    .map((item) => item.index);
}

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (!fs.existsSync(options.source)) throw new Error(`源目录不存在: ${options.source}`);
  fs.mkdirSync(options.output, { recursive: true });
  const fontInfo = initializeSkia();
  const matchedFiles = options.match
    ? walkFiles(options.source).filter((filePath) => path.basename(filePath).includes(options.match))
    : walkFiles(options.source);
  const files = matchedFiles.slice(0, options.limit);
  const corpusEntries = [];

  for (const [fileIndex, sourcePath] of files.entries()) {
    const sourceBytes = fs.readFileSync(sourcePath);
    const sourceHash = sha256(sourceBytes);
    const sourceId = sourceHash.slice(0, 16);
    const sourceDir = path.join(options.output, sourceId);
    fs.mkdirSync(sourceDir, { recursive: true });
    console.log(`[${fileIndex + 1}/${files.length}] 解析 ${path.basename(sourcePath)}`);

    try {
      const importSettings = new alphaTab.Settings();
      const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(new Uint8Array(sourceBytes), importSettings);
      const stringedTrackIndexes = preferredStringedTrackIndexes(score);
      const renderTrackIndexes = stringedTrackIndexes;
      const renders = [];
      const renderErrors = [];

      for (const trackIndex of renderTrackIndexes) {
        console.log(`  渲染轨道 ${trackIndex}: ${score.tracks[trackIndex].name}`);
        try {
          const rendered = await renderTrack(score, trackIndex, options, fontInfo);
          const stem = `track-${String(trackIndex).padStart(2, '0')}-${options.profile}`;
          const imagePath = path.join(sourceDir, `${stem}.png`);
          const labelPath = path.join(sourceDir, `${stem}.labels.json`);
          fs.writeFileSync(imagePath, rendered.png);
          writeJson(labelPath, {
            schema_version: 2,
            source_id: sourceId,
            track_index: trackIndex,
            track: trackSummary(score.tracks[trackIndex]),
            profile: options.profile,
            image: { file: path.basename(imagePath), ...rendered.finalSize },
            ...rendered.labels,
          });
          renders.push({
            track_index: trackIndex,
            image: path.relative(options.output, imagePath),
            labels: path.relative(options.output, labelPath),
            ...rendered.finalSize,
            beat_count: rendered.labels.beat_count,
            note_box_count: rendered.labels.note_box_count,
            rendered_partials: rendered.renderedPartials,
          });
          if (!options.allTracks) break;
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          renderErrors.push({ track_index: trackIndex, error: message });
          console.warn(`    轨道失败: ${message}`);
        }
      }
      if (renders.length === 0) {
        const detail = renderErrors.map((item) => `t${item.track_index}: ${item.error}`).join('; ');
        throw new Error(detail || '没有可渲染的六线谱轨道');
      }

      const entry = {
        schema_version: 1,
        source_id: sourceId,
        source_path: path.resolve(sourcePath),
        source_sha256: sourceHash,
        source_size: sourceBytes.byteLength,
        extension: path.extname(sourcePath).toLowerCase(),
        title: score.title,
        sub_title: score.subTitle,
        artist: score.artist,
        album: score.album,
        tempo: score.tempo,
        master_bar_count: score.masterBars.length,
        tracks: score.tracks.map(trackSummary),
        renders,
        render_errors: renderErrors,
        status: 'ok',
      };
      writeJson(path.join(sourceDir, 'source.json'), entry);
      corpusEntries.push(entry);
    } catch (error) {
      const entry = {
        schema_version: 1,
        source_id: sourceId,
        source_path: path.resolve(sourcePath),
        source_sha256: sourceHash,
        source_size: sourceBytes.byteLength,
        extension: path.extname(sourcePath).toLowerCase(),
        status: 'error',
        error: error instanceof Error ? error.message : String(error),
      };
      writeJson(path.join(sourceDir, 'source.json'), entry);
      corpusEntries.push(entry);
      console.error(`  失败: ${entry.error}`);
    }
  }

  const manifestPath = path.join(options.output, 'manifest.jsonl');
  fs.writeFileSync(manifestPath, corpusEntries.map((entry) => JSON.stringify(entry)).join('\n') + '\n', 'utf8');
  const successful = corpusEntries.filter((entry) => entry.status === 'ok');
  console.log(`完成: ${successful.length}/${corpusEntries.length} 个文件，输出 ${options.output}`);
}

await main();
