//
// Copyright © 2023 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

use clap::Parser;
use serde_json::{json, Map, Value};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, BufWriter, Seek},
    path::{Path, PathBuf},
};
use xml::{
    common::{Position, TextPosition},
    name::OwnedName,
    reader::XmlEvent,
    EventReader,
};

// List of color names acceptable for the "cname" property.
// Retrieved from:
// https://github.com/catapult-project/catapult/blob/11513e359cd60e369bbbd1f4f2ef648c1bccabd0/tracing/tracing/base/color_scheme.html
// but filtered to remove colors which are not usable (e.g. too similar to background).
const COLORS: [&'static str; 29] = [
    "thread_state_uninterruptible",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_unknown",
    "background_memory_dump",
    "detailed_memory_dump",
    "vsync_highlight_color",
    "generic_work",
    "good",
    "bad",
    "grey",
    "yellow",
    "olive",
    "rail_response",
    "rail_animation",
    "rail_idle",
    "rail_load",
    "startup",
    "heap_dump_stack_frame",
    "heap_dump_object_type",
    "heap_dump_child_node_arrow",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "cq_build_abandoned",
    "cq_build_attempt_runnig",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
];

fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

// fn timestamp_to_string(timestamp) {
//     // The python datetime object stores only microsecond precision, but the timestamps we have are in nanoseconds.
//     // Therefore we handle the fractional seconds separately so that we don"t lose the last 3 decimal places.
//     // Whole seconds part
//     a = datetime.datetime.fromtimestamp(timestamp // int(1e9))
//     // Fractional seconds
//     b = timestamp % int(1e9)
//     return "{}.{:09}".format(a, b)
// }

/// Stores information about an agent or command XML element.
#[derive(Clone, Debug)]
struct XmlElement {
    name: String,
    child_element_values: HashMap<String, String>,
    text_representation: String,
}

#[derive(Default)]
struct CommandList {
    commands: Vec<XmlElement>,
    current_idx_per_filter: HashMap<String, usize>,
}
impl CommandList {
    // We don't report the command index in the profiling data, so we have to reconstruct this.
    // We know that commands are started by the firmware in order, so we know that e.g. the second
    // START_MCE_STRIPE command we see in the profiling trace must correspond to the second START_MCE_STRIPE
    // command in the (MCE) command list. This function handles that logic.
    fn advance(&mut self, filter_id: &str, command_name: &str) -> usize {
        if !self.current_idx_per_filter.contains_key(filter_id) {
            self.current_idx_per_filter.insert(filter_id.to_string(), 0);
        }

        let mut idx = self.current_idx_per_filter[filter_id];

        // Skip past anything that doesn"t match the filter
        while self
            .commands
            .get(idx)
            .expect("Attempted to increment past end of command stream. Something has gone wrong")
            .name
            != command_name
        {
            idx += 1;
        }

        // Return the index of the first thing that matches the filter
        let result = idx;

        // Move on once for next time
        idx += 1;
        *self.current_idx_per_filter.get_mut(filter_id).unwrap() = idx;

        result
    }
}

struct Agent {
    xml: XmlElement,
    start_timestamp: Option<u64>,
    end_timestamp: Option<u64>,
}

/// Extract and return the list of agents and commands from the command stream.
fn parse_command_stream(
    command_stream_filename: &Path,
) -> (
    Vec<Agent>,
    CommandList,
    CommandList,
    CommandList,
    CommandList,
) {
    // All the Rust XML libraries I have checked are missing some needed features, like:
    //  * Ergonomic ownership semantics (e.g. not having to have lifetime parameters everywhere)
    //  * Parsing into a DOM
    //  * Supporting comments (even if they're skipped)
    //  * Getting a text representation of an element like .outerHTML
    // Current solution is to use a library without a DOM (just a streaming parser) and extract the bits we need manually :(

    let mut command_stream_file = BufReader::new(
        std::fs::File::open(command_stream_filename).expect("Failed to open command stream"),
    );
    // Read all the lines up-front, as we'll need these for generating text representations of the XML elements later
    let mut command_stream_lines = vec![];
    loop {
        let mut buffer = String::new();
        match command_stream_file.read_line(&mut buffer) {
            Ok(0) => break,
            Ok(n) => command_stream_lines.push(buffer[0..n - 1].to_string()), // Trim the trailing newline
            Err(e) => panic!("Error reading lines from command stream: {e}"),
        }
    }
    command_stream_file.rewind().unwrap();

    let mut parser = EventReader::new(command_stream_file);

    let mut agents = vec![];
    let mut dma_rd_commands = CommandList::default();
    let mut dma_wr_commands = CommandList::default();
    let mut mce_commands = CommandList::default();
    let mut ple_commands = CommandList::default();

    /// An XML element which we have seen the opening tag for, but not the closing tag.
    struct PendingXmlElement {
        name: OwnedName,
        child_element_values: HashMap<String, String>,
        start_pos: TextPosition,
    }
    // Converts a PendingXmlElement into a finished proper XmlElement, now that we have seen the closing tag
    let finish_pending = |p: PendingXmlElement, end_pos: TextPosition| -> XmlElement {
        let mut text_representation = String::new();
        // Reconstruct the text representation of the XML element, by joining all the lines.
        // Also strip off leading indentation
        for line in p.start_pos.row..=end_pos.row {
            text_representation += command_stream_lines[line as usize]
                .split_at(p.start_pos.column as usize)
                .1;
            text_representation += "\n";
        }

        XmlElement {
            name: p.name.local_name,
            child_element_values: p.child_element_values,
            text_representation,
        }
    };

    let mut element_name_stack: Vec<String> = vec![];
    let mut pending_elements_stack = vec![];

    loop {
        let event = parser.next().expect("Error parsing command stream XML");
        match event {
            XmlEvent::StartElement { name, .. } => {
                pending_elements_stack.push(PendingXmlElement {
                    name: name.clone(),
                    child_element_values: HashMap::new(),
                    start_pos: parser.position(),
                });
                element_name_stack.push(name.local_name);
            }
            XmlEvent::Characters(data) => {
                // Record the values inside child elements, as this is how we store fields of agents/commands in the command stream
                let n = pending_elements_stack.len();
                if let Some(grandparent) = pending_elements_stack.get_mut(n - 2) {
                    grandparent
                        .child_element_values
                        .insert(element_name_stack.last().unwrap().clone(), data);
                }
            }
            XmlEvent::EndElement { .. } => {
                element_name_stack.pop();
                let pending = pending_elements_stack.pop().unwrap();

                if element_name_stack.last() == Some(&"AGENTS".to_string()) {
                    agents.push(Agent {
                        xml: finish_pending(pending, parser.position()),
                        start_timestamp: None,
                        end_timestamp: None,
                    })
                } else if element_name_stack.last() == Some(&"DMA_RD_COMMANDS".to_string()) {
                    dma_rd_commands
                        .commands
                        .push(finish_pending(pending, parser.position()));
                } else if element_name_stack.last() == Some(&"DMA_WR_COMMANDS".to_string()) {
                    dma_wr_commands
                        .commands
                        .push(finish_pending(pending, parser.position()));
                } else if element_name_stack.last() == Some(&"MCE_COMMANDS".to_string()) {
                    mce_commands
                        .commands
                        .push(finish_pending(pending, parser.position()));
                } else if element_name_stack.last() == Some(&"PLE_COMMANDS".to_string()) {
                    ple_commands
                        .commands
                        .push(finish_pending(pending, parser.position()));
                }
            }
            XmlEvent::EndDocument => break,
            _ => (),
        }
    }

    (
        agents,
        dma_rd_commands,
        dma_wr_commands,
        mce_commands,
        ple_commands,
    )
}

/// Gets a tuple describing how this timeline entry should be displayed. The tuple contains:
///     - section (process) name
///     - row (thread) name
///     - event name (text to place on the event bar)
///     - args (data to be shown at the bottom when the event is selected)
///     - color
/// Note we prefix the process/thread names with a/b/c etc. to force a specific order
/// (as they get displayed alphabetically).
fn process_timeline_event_start_or_instant(
    entry: &Map<String, Value>,
    agents: &[Agent],
    dma_rd_commands: &mut CommandList,
    dma_wr_commands: &mut CommandList,
    mce_commands: &mut CommandList,
    ple_commands: &mut CommandList,
    mce_bank: &mut u32,
) -> (
    String,
    String,
    String,
    serde_json::Map<String, Value>,
    String,
) {
    let mut args = serde_json::Map::new();
    args.insert("entry".to_string(), Value::Object(entry.clone()));
    let metadata_category = entry["metadata_category"].as_str().unwrap();

    let mut handle_command_and_agent = |command_list: &mut CommandList,
                                        command_name: &str|
     -> (usize, XmlElement, usize, XmlElement) {
        let command_idx = command_list.advance(metadata_category, command_name);
        let command_xml = command_list.commands[command_idx].clone();
        let agent_id: usize = command_xml
            .child_element_values
            .get("AGENT_ID")
            .unwrap()
            .parse()
            .unwrap();
        let agent_xml = agents[agent_id].xml.clone();
        args.insert("command_idx".to_string(), command_idx.into());
        args.insert(
            "command_xml".to_string(),
            command_xml.text_representation.clone().into(),
        );
        args.insert("agent_id".to_string(), agent_id.into());
        args.insert(
            "agent_xml".to_string(),
            agent_xml.text_representation.clone().into(),
        );

        (agent_id, agent_xml, command_idx, command_xml)
    };

    match metadata_category {
        "FirmwareInference" => (
            "b) Command Stream".to_string(),
            "a) Inference".to_string(),
            "Inference".to_string(),
            args,
            "".to_string(),
        ),
        "FirmwareUpdateProgress" => (
            "c) NCU MCU".to_string(),
            "a) Events".to_string(),
            "UpdateProgress".to_string(),
            args,
            "".to_string(),
        ),
        "FirmwareWfe" => (
            "c) NCU MCU".to_string(),
            "a) Events".to_string(),
            "WFE".to_string(),
            args,
            "".to_string(),
        ),
        "FirmwareDmaReadSetup" => {
            let (agent_id, _, _, _) = handle_command_and_agent(dma_rd_commands, "DMA_COMMAND");
            // The agent in the command stream should specify if this is weights, ple, ifm etc.
            let agent_type = agents[agent_id].xml.name.clone();
            args.insert("agent_type".to_string(), agent_type.clone().into());
            (
                "c) NCU MCU".to_string(),
                "d) DMA stripe setup".to_string(),
                agent_type,
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwareDmaRead" => {
            let (agent_id, _, _, command_xml) =
                handle_command_and_agent(dma_rd_commands, "DMA_COMMAND");
            // The agent in the command stream should specify if this is weights, ple, ifm etc.
            let agent_type = agents[agent_id].xml.name.clone();
            args.insert("agent_type".to_string(), agent_type.clone().into());
            // The hardware ID can be extracted from the command stream
            let hardware_id = u32::from_str_radix(
                &command_xml.child_element_values.get("DMA_CMD").unwrap()[2..],
                16,
            )
            .unwrap()
                & 0b111;
            args.insert("hardware_id".to_string(), hardware_id.into());
            (
                "d) DMA".to_string(),
                format!("a) DMA Load {}", hardware_id),
                agent_type,
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwareDmaWriteSetup" => {
            let (agent_id, _, _, _) = handle_command_and_agent(dma_wr_commands, "DMA_COMMAND");
            (
                "c) NCU MCU".to_string(),
                "d) DMA stripe setup".to_string(),
                "OFM_STREAMER".to_string(),
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwareDmaWrite" => {
            let (agent_id, _, _, command_xml) =
                handle_command_and_agent(dma_wr_commands, "DMA_COMMAND");
            // The hardware ID can be extracted from the command stream
            let hardware_id = u32::from_str_radix(
                &command_xml.child_element_values.get("DMA_CMD").unwrap()[2..],
                16,
            )
            .unwrap()
                & 0b111;
            args.insert("hardware_id".to_string(), hardware_id.into());
            (
                "d) DMA".to_string(),
                format!("a) DMA Save {}", hardware_id),
                "OFM_STREAMER".to_string(),
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwareMceStripeSetup" => {
            let (agent_id, _, _, _) =
                handle_command_and_agent(mce_commands, "PROGRAM_MCE_STRIPE_COMMAND");
            (
                "c) NCU MCU".to_string(),
                "c) MCE stripe setup".to_string(),
                "MCE stripe setup".to_string(),
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwareMceStripe" => {
            let (agent_id, agent_xml, _, _) =
                handle_command_and_agent(mce_commands, "START_MCE_STRIPE_COMMAND");
            *mce_bank = (*mce_bank + 1) % 2;
            // Get operation (depthwise vs. conv etc.) from the command stream
            let operation = agent_xml.child_element_values.get("MCE_OP_MODE").unwrap();
            (
                "f) MCE".to_string(),
                format!("a) MCE bank {}", mce_bank),
                format!("{}", operation),
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwarePleStripeSetup" => {
            let (agent_id, _, _, _) =
                handle_command_and_agent(ple_commands, "START_PLE_STRIPE_COMMAND");
            (
                "c) NCU MCU".to_string(),
                "c) PLE stripe setup".to_string(),
                format!("PLE stripe setup"),
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwarePleStripe" => {
            let (agent_id, agent_xml, _, _) =
                handle_command_and_agent(ple_commands, "START_PLE_STRIPE_COMMAND");
            // Get PLE kernel ID from the command stream
            let ple_kernel_id = agent_xml.child_element_values.get("PLE_KERNEL_ID").unwrap();
            (
                "g) PLE".to_string(),
                "a) PLE".to_string(),
                format!("{}", ple_kernel_id),
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwareUdma" => {
            let (agent_id, _, _, _) =
                handle_command_and_agent(ple_commands, "LOAD_PLE_CODE_INTO_PLE_SRAM_COMMAND");
            (
                "e) UDMA".to_string(),
                "a) UDMA".to_string(),
                format!("UDMA"),
                args,
                COLORS[agent_id % COLORS.len()].to_string(),
            )
        }
        "FirmwareLabel" => {
            let label = entry["metadata"]["label"].as_str().unwrap().to_string();
            args.insert("label".to_string(), label.clone().into());
            (
                "c) NCU MCU".to_string(),
                "g) LABELS".to_string(),
                label,
                args,
                COLORS[0].to_string(),
            )
        }
        "InferenceLifetime" => (
            "a) Driver Library".to_string(),
            format!("a) Inference"),
            format!("Inference"),
            args,
            "".to_string(),
        ),
        "BufferLifetime" => (
            "a) Driver Library".to_string(),
            format!("b) Buffer {}", entry["id"]),
            format!("Buffer {}", entry["id"]),
            args,
            "".to_string(),
        ),
        x => {
            panic!("Unknown metadata category: {x}");
        }
    }
}

/// Gets a tuple of describing how this counter entry should be displayed. The tuple contains:
///   - section (process) name
///   - row (thread) name
///   - event name (text to place on the event bar)
///   - args (data to be shown at the bottom when the event is selected)
///   - color
fn process_counter_entry(
    entry: &Map<String, Value>,
) -> (
    String,
    String,
    String,
    serde_json::Map<String, Value>,
    String,
) {
    let name = entry["counter_name"].as_str().unwrap();
    let value = entry["counter_value"].as_u64().unwrap();

    let mut args = serde_json::Map::new();
    args.insert(name.to_string(), value.into());

    (
        "z) Counters".to_string(),
        "<unused>".to_string(),
        name.to_string(),
        args,
        "".to_string(),
    )
}

// fn create_bar_events(
//     name: String,
//     timestamp_begin: u64,
//     timestamp_end: u64,
//     process_id: u64,
//     thread_id: u64,
//     args: Map<String, Value>,
// ) -> [Value; 2] {
//     let mut begin_event = json!({
//         "name": name,
//         "ph": "B",
//         "ts": timestamp_begin,
//         "pid": process_id,
//         "tid": thread_id,
//     });
//     begin_event["args"] = Value::Object(args);

//     let mut end_event = begin_event.clone();
//     end_event["ph"] = "E".into();
//     end_event["ts"] = timestamp_end.into();

//     [begin_event, end_event]
// }

fn process_finalize(
    mut data: Vec<Value>,
    agents: &[Agent],
    process_names: &mut HashMap<u64, String>,
    thread_names: &mut HashMap<(u64, u64), String>,
    add_timeline_bars: bool,
) -> Vec<Value> {
    // Add events to show the start and end of each agent
    let process_name = "b) Command Stream".to_string();
    let pid = hash_string(&process_name);
    process_names.insert(pid, process_name);
    for (agent_idx, agent) in agents.iter().enumerate() {
        if agent.start_timestamp.is_none() {
            continue;
        }

        // Format command number as fixed-width (e.g. 003) so that it
        // is in the correct order when Chrome sorts alphabetically
        let thread_name = format!("b) Agent {:04} ({})", agent_idx, agent.xml.name);
        let tid = hash_string(&thread_name);
        thread_names.insert((pid, tid), thread_name);

        let begin_event = json!({
            "name": format!("Agent {} ({})", agent_idx, agent.xml.name),
            "ph": "B",
            "ts": agent.start_timestamp.unwrap(),
            "pid": pid,
            "tid": tid,
            "args": { "agent_xml": agents[agent_idx].xml.text_representation },
        });
        data.push(begin_event.clone());

        let mut end_event = begin_event.clone();
        end_event["ph"] = "E".into();
        end_event["ts"] = agent.end_timestamp.unwrap().into();
        data.push(end_event);
    }

    // Add a fake "End" event for any timeline events which we didn"t find an end for. This might be because
    // for example a buffer was still alive when the profiling data was dumped. If we don"t add an end event ourselves,
    // Chrome displays these more like an instantaneous event, which can be confusing.
    // Also do some validation
    let max_timestamp = data
        .iter()
        .map(|e| e.as_object().unwrap().get("ts").unwrap().as_u64().unwrap())
        .max();
    let mut begin_events = HashMap::<(u64, u64), Value>::new();
    for entry in &data {
        let key = (
            entry["pid"].as_u64().unwrap(),
            entry["tid"].as_u64().unwrap(),
        ); // pid and tid should uniquely identify the timeline event (which should have an end too)
        if entry["ph"] == "B" {
            if begin_events.contains_key(&key) {
                println!(
                    "Warning: Begin event twice in a row before an End: {}",
                    entry
                );
            } else {
                begin_events.insert(key, entry.clone());
            }
        } else if entry["ph"] == "E" {
            if begin_events.contains_key(&key) {
                // Check timestamp for end event is after begin
                if entry["ts"].as_u64().unwrap()
                    < begin_events.get(&key).unwrap()["ts"].as_u64().unwrap()
                {
                    panic!("End event ({}) timestamp is before beginning ({})! Chrome won't display this", entry, begin_events.get(&key).unwrap());
                }

                begin_events.remove(&key);
            } else {
                println!(
                    "Warning: End event does not have corresponding Begin: {}",
                    entry
                )
            }
        }
    }
    for begin_event in begin_events.values_mut() {
        begin_event["name"] =
            (begin_event["name"].as_str().unwrap().to_string() + " (NOT ENDED)").into();
        let mut end_event = begin_event.clone();
        end_event["ph"] = "E".into();
        end_event["ts"] = max_timestamp.into();
        data.push(end_event);
    }

    if add_timeline_bars {
        panic!("add_timeline_bars not implemented!");
        //     // Add some bars to display a timeline at the top. Because we don"t convert our timestamps to the expected Chrome
        //     // units (see code comment in process_entry for explanation), the built-in timeline is confusing so we add our own.
        //     // These bars can also be useful for zooming to a fixed scale, for comparing two traces.
        //     timeline_pid = hash("timeline")
        //     process_names[timeline_pid] = "0) Timeline"
        //     ms_tid = hash("ms")
        //     thread_names[(timeline_pid, ms_tid)] = "b) ms"
        //     s_tid = hash("s")
        //     thread_names[(timeline_pid, s_tid)] = "a) s"
        //     oldest_timestamp = min(map(lambda d: d["ts"], data))
        //     newest_timestamp = max(map(lambda d: d["ts"], data))
        //     for t in range(oldest_timestamp, newest_timestamp, 1000*1000):
        //         // Add a millisecond marker
        //         ms_within_s = ((t - oldest_timestamp) / int(1e6)) % int(1e3)
        //         // Chrome automatically offsets everything so the earliest timestamp is zeroed, therefore losing the absolute
        //         // time. Add this information as metadata as it may be useful.
        //         args = { "wall_clock": timestamp_to_string(t) }
        //         data.extend(create_bar_events("{:,} ms".format(ms_within_s), t, t + 1000*1000, timeline_pid, ms_tid, args))

        //         // Add a second markers, every 1000 milliseconds
        //         if ms_within_s == 0:
        //             s = int((t - oldest_timestamp) / 1e9)
        //             data.extend(create_bar_events("{:,} s".format(s), t, t + 1000*1000*1000, timeline_pid, s_tid, args))

        //     freq = int(5e6)
        //     cycles_per_bar = 1000
        //     nanosecs_per_bar = round(1e9 / freq * cycles_per_bar)
        //     cycles_tid = hash("cycles")
        //     thread_names[(timeline_pid, cycles_tid)] = "c) cycles (assuming 5MHz)"
        //     for t in range(oldest_timestamp, newest_timestamp, nanosecs_per_bar):
        //         cycles = (t - oldest_timestamp) * freq // int(1e9)
        //         // Chrome automatically offsets everything so the earliest timestamp is zeroed, therefore losing the absolute
        //         // time. Add this information as metadata as it may be useful.
        //         args = { "wall_clock": timestamp_to_string(t) }
        //         data.extend(
        //             create_bar_events("{:,} cycles".format(cycles), t, t + nanosecs_per_bar, timeline_pid, cycles_tid, args))
    }

    // Append the metadata which gives a name to each process
    for (pid, name) in process_names {
        let metadata = json!({
            "name": "process_name", "ph": "M", "pid": pid, "args": {"name": name}
        });
        data.push(metadata);
    }
    // Append the metadata which gives a name to each thread
    for ((pid, tid), name) in thread_names {
        let metadata = json!({
            "name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": name}
        });
        data.push(metadata)
    }

    return data;
}

fn process_entry(
    entry: &Map<String, Value>,
    in_progress_events: &mut HashMap<u64, (u64, u64, Option<usize>)>,
    process_names: &mut HashMap<u64, String>,
    thread_names: &mut HashMap<(u64, u64), String>,
    agents: &mut [Agent],
    dma_rd_commands: &mut CommandList,
    dma_wr_commands: &mut CommandList,
    mce_commands: &mut CommandList,
    ple_commands: &mut CommandList,
    mce_bank: &mut u32,
) -> Value {
    let entry_type = entry["type"].as_str().unwrap();
    let timestamp = entry["timestamp"].as_u64().unwrap();
    // Timestamps from the driver library"s json dump are in nanoseconds, but the Chrome Trace View format uses
    // microseconds. It only support whole numbers though, so we DELIBERATELY DONT convert correctly here so that
    // we don"t lose precision. The downside of this is that the timeline in Chrome will show everything taking
    // 1000x longer than it did in reality.
    let chrome_ts = timestamp;

    if entry_type == "TimelineEventEnd" {
        // All we need is the PID and TID from the start event
        let (pid, tid, agent_id) = in_progress_events
            .get(&entry["id"].as_u64().unwrap())
            .unwrap();

        // Update agent lifetime
        if let Some(agent_id) = agent_id {
            agents[*agent_id].end_timestamp = Some(std::cmp::max(
                agents[*agent_id].end_timestamp.unwrap_or(0),
                timestamp,
            ));
        }

        return json!({
            "ph": "E",
            "ts": chrome_ts,
            "pid": pid,
            "tid": tid,
        });
    }

    let (ph, process_name, thread_name, name, args, color) = if entry_type == "TimelineEventStart" {
        let (process_name, thread_name, name, args, color) =
            process_timeline_event_start_or_instant(
                entry,
                agents,
                dma_rd_commands,
                dma_wr_commands,
                mce_commands,
                ple_commands,
                mce_bank,
            );

        ("B", process_name, thread_name, name, args, color)
    } else if entry_type == "TimelineEventInstant" {
        let (process_name, thread_name, name, args, color) =
            process_timeline_event_start_or_instant(
                entry,
                agents,
                dma_rd_commands,
                dma_wr_commands,
                mce_commands,
                ple_commands,
                mce_bank,
            );

        ("I", process_name, thread_name, name, args, color)
    } else if entry_type == "CounterSample" {
        let (process_name, thread_name, name, args, color) = process_counter_entry(entry);
        ("C", process_name, thread_name, name, args, color)
    } else {
        panic!("Unknown entry_type ({})", entry_type)
    };

    // Each section (process) needs a unique numerical ID.
    let pid = hash_string(&process_name);
    // Store the mapping from process ID to name, so we can list it in the JSON file at the end
    process_names.insert(pid, process_name);

    // Each row (thread) needs a unique numerical ID.
    // Store the mapping from (process ID, thread ID) to name, so we can list it in the JSON file at the end
    let tid = hash_string(&thread_name);
    thread_names.insert((pid, tid), thread_name);

    // Construct the JSON object
    let result = json!({
        "name": name,
        "ph": ph,
        "ts": chrome_ts,
        "pid": pid,
        "tid": tid,
        "args": args,
        "cname": color
    });

    if entry_type == "TimelineEventStart" {
        // Update agent lifetime
        let mut agent_id_maybe = None;
        if let Some(agent_id) = args.get("agent_id") {
            let agent_id = agent_id.as_u64().unwrap() as usize;
            if agents[agent_id].start_timestamp.is_none() {
                agents[agent_id].start_timestamp = Some(timestamp);
            }
            agent_id_maybe = Some(agent_id);
        }

        in_progress_events.insert(entry["id"].as_u64().unwrap(), (pid, tid, agent_id_maybe));
    }

    result
}

fn process_json(
    entries: &[Value],
    agents: &mut [Agent],
    dma_rd_commands: &mut CommandList,
    dma_wr_commands: &mut CommandList,
    mce_commands: &mut CommandList,
    ple_commands: &mut CommandList,
    add_timeline_bars: bool,
) -> Vec<Value> {
    // This will be built up with the list of trace objects to save to the JSON file
    let mut result = vec![];
    // This will be built up with a map from process IDs to names and then added to the end of the JSON file.
    let mut process_names = HashMap::new();
    // Will be built up with a map of (process ID, thread ID) to thread names and then added to the end of the JSON file
    let mut thread_names = HashMap::new();

    let mut in_progress_events = HashMap::new();

    let mut mce_bank = 1;

    // Convert each line that has an entry
    for entry in entries {
        // Extract the fields for the entry
        let output_json_object = process_entry(
            entry.as_object().unwrap(),
            &mut in_progress_events,
            &mut process_names,
            &mut thread_names,
            agents,
            dma_rd_commands,
            dma_wr_commands,
            mce_commands,
            ple_commands,
            &mut mce_bank,
        );

        result.push(output_json_object);
    }

    process_finalize(
        result,
        agents,
        &mut process_names,
        &mut thread_names,
        add_timeline_bars,
    )
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Filename to read profiling entries dumped out by the driver library
    #[arg(short, long, default_value = "profiling.json")]
    profiling_entries: PathBuf,

    /// Filename to read the command stream from, dumped out by the driver library
    #[arg(short, long, default_value = "CommandStream_.xml")]
    command_stream: PathBuf,

    /// Filename to write the output JSON to
    #[arg(short, long, default_value = "trace.json")]
    output: PathBuf,

    /// Add bars displayed at the top of the trace which show a timeline.
    /// Because the timestamps in the produced trace file are not displayed correctly in Chrome,
    /// the built-in timeline is confusing so this adds an alternative.
    /// These bars can also be useful for zooming to a fixed scale, for comparing two traces
    #[arg(short, long, default_value_t = false)]
    add_timeline_bars: bool,
}

fn main() {
    let args = Args::parse();

    // Try to extract the commands from the command stream
    let (mut agents, mut dma_rd_commands, mut dma_wr_commands, mut mce_commands, mut ple_commands) =
        parse_command_stream(&args.command_stream);

    let input_file = BufReader::new(
        std::fs::File::open(args.profiling_entries).expect("Failed to open input JSON file"),
    );
    let input_json: Value =
        serde_json::from_reader(input_file).expect("Failed to parse input JSON");

    let output_json = process_json(
        input_json.as_array().expect("Invalid json"),
        &mut agents,
        &mut dma_rd_commands,
        &mut dma_wr_commands,
        &mut mce_commands,
        &mut ple_commands,
        args.add_timeline_bars,
    );
    let output_file = BufWriter::new(
        std::fs::File::create(&args.output).expect("Failed to create output JSON file"),
    );
    serde_json::to_writer_pretty(output_file, &output_json).expect("Failed to save output JSON");

    println!(
        "Saved {} events to {}",
        output_json.len(),
        args.output.display()
    );
}
