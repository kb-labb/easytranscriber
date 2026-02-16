library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(patchwork)

my_theme <- theme(
    # Add white space around plot
    plot.margin = margin(t = 10, r = 10, b = 14, l = 10),
    # Push x-axis title and labels further down
    axis.title.x = element_text(margin = margin(t = 12)),
    axis.text.x = element_text(
        # angle = 6,
        margin = margin(t = 2)
    ),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.background = element_rect(fill = "#fdf6e3"),
    legend.background = element_rect(fill = "#f5e6c8", color = NA),
    legend.key.spacing.y = unit(2, "pt"),
    legend.key = element_rect(fill = "grey20", color = NA),
    plot.background = element_rect(
        fill = "#fdf6e3",
        color = "#fdf6e3",
        linewidth = 2
    )
)

df_bench <- read_csv("benchmarks/results.csv")

df <- df_bench %>%
    group_by(bench, pipeline, library, vad, backend) %>%
    summarise(
        mean_seconds = mean(seconds),
        cpu = first(cpu),
        gpu = first(gpu),
        num_beams = first(num_beams),
        .groups = "drop"
    )


df <- df %>%
    filter(backend != "huggingface")

df <- df %>%
    mutate(lib_vad = paste(library, vad, sep = "-"))

df <- df %>%
    group_by(bench) %>%
    mutate(speedup = mean_seconds[library == "whisperx"] / mean_seconds) %>%
    ungroup()

p1 <- ggplot(
    df %>% filter(bench == "bench1"),
    aes(x = lib_vad, fill = library, y = speedup)
) +
    geom_bar(stat = "identity", position = "dodge", color = "grey20") +
    geom_hline(
        yintercept = 1,
        linetype = "dashed",
        color = "#657b83",
        size = 0.7
    ) +
    geom_text(
        aes(label = sprintf("%.2fx", speedup)),
        vjust = -0.5,
        size = 3.5
    ) +
    scale_fill_manual(
        values = c(
            easytranscriber = "#5ebbf7",
            whisperx = "#f28e2b"
        )
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.07))) +
    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
    labs(title = "CPU: i7-9700K @ 3.60 GHz \nGPU: RTX 3090") +
    theme_bw(base_size = 13) +
    my_theme +
    labs(x = "Library + VAD combination", y = "Speedup", fill = "Library")

# ggsave(
#     "benchmarks/plots/bench1_speedup.png",
#     p1,
#     width = 6,
#     height = 6,
#     dpi = 300
# )

p2 <- ggplot(
    df %>% filter(bench == "bench4" & pipeline != "ct2_dataloader"),
    aes(x = lib_vad, fill = library, y = speedup)
) +
    geom_bar(stat = "identity", position = "dodge", color = "grey20") +
    geom_hline(
        yintercept = 1,
        linetype = "dashed",
        color = "#657b83",
        size = 0.7
    ) +
    geom_text(
        aes(label = sprintf("%.2fx", speedup)),
        vjust = -0.5,
        size = 3.5
    ) +
    scale_fill_manual(
        values = c(
            easytranscriber = "#5ebbf7",
            whisperx = "#f28e2b"
        )
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.07))) +
    scale_x_discrete(guide = guide_axis(n.dodge = 1)) +
    labs(title = "CPU: AMD Rome 7742 @ 2.25 GHz \nGPU: A100 (40GB)") +
    theme_bw(base_size = 13) +
    my_theme +
    theme(
        axis.text.x = element_blank(),
        # axis.text.x = element_text(angle = -7, hjust = 0.5, margin = margin(t = 0))
    ) +
    labs(x = "VAD backend = Pyannote", y = "Speedup", fill = "Library")

# ggsave(
#     "benchmarks/plots/bench4_speedup.png",
#     p2,
#     width = 6,
#     height = 6,
#     dpi = 300
# )

p3 <- ggplot(
    df %>% filter(bench == "bench3"),
    aes(x = lib_vad, fill = library, y = speedup)
) +
    geom_bar(stat = "identity", position = "dodge", color = "grey20") +
    geom_hline(
        yintercept = 1,
        linetype = "dashed",
        color = "#657b83",
        size = 0.7
    ) +
    geom_text(
        aes(label = sprintf("%.2fx", speedup)),
        vjust = -0.5,
        size = 3.5
    ) +
    scale_fill_manual(
        values = c(
            easytranscriber = "#5ebbf7",
            whisperx = "#f28e2b"
        )
    ) +
    scale_y_continuous(
        expand = expansion(mult = c(0, 0.07)),
        limits = c(0, 2.02)
    ) +
    scale_x_discrete(guide = guide_axis(n.dodge = 1)) +
    labs(title = "CPU: Ryzen 5950X @ 3.4 GHz \nGPU: RTX A5000 (Ada)") +
    theme_bw(base_size = 13) +
    my_theme +
    theme(
        axis.text.x = element_blank(),
        # axis.text.x = element_text(angle = -7, hjust = 0.5, margin = margin(t = 0))
    ) +
    labs(x = "VAD backend = Pyannote", y = "Speedup", fill = "Library")

# ggsave(
#     "benchmarks/plots/bench3_speedup.png",
#     p3,
#     width = 6,
#     height = 6,
#     dpi = 300
# )

combined <- (p1 + theme(legend.position = "none")) +
    (p2 + theme(legend.position = "none")) +
    p3 &
    plot_annotation(
        title = "Benchmark easytranscriber vs WhisperX",
        theme = theme(
            plot.title = element_text(
                size = 20,
                face = "bold",
                hjust = 0.5,
                margin = margin(b = 10)
            ),
            plot.background = element_rect(
                fill = "#fdf6e3",
                color = "grey40",
                linewidth = 1
            )
        )
    )
ggsave(
    "benchmarks/plots/all_speedup.png",
    combined,
    width = 15,
    height = 5.5,
    dpi = 300,
)

# markdown table output of df_bench
df_bench %>%
    arrange(bench, pipeline, library, vad, backend) %>%
    knitr::kable()
