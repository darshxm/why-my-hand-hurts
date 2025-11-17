const { createApp, reactive, ref, nextTick, onMounted } = Vue;

const charts = {
  freq: null,
  duration: null,
};

async function fetchJson(path, query = "") {
  const url = query ? `${path}?${query}` : path;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Request failed (${res.status}): ${text || res.statusText}`);
  }
  return res.json();
}

function renderBar(chartRef, { labels, data, label, color }) {
  if (charts[chartRef]) {
    charts[chartRef].destroy();
  }
  const ctx = document.getElementById(chartRef + "Chart");
  if (!ctx) return;
  charts[chartRef] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label,
          data,
          backgroundColor: color,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: "#cfd7e6" },
          grid: { display: false },
        },
        y: {
          ticks: { color: "#cfd7e6" },
          grid: { color: "#1f2c43" },
        },
      },
      plugins: {
        legend: { labels: { color: "#cfd7e6" } },
      },
    },
  });
}

createApp({
  setup() {
    const filters = reactive({ app: "", window: "" });
    const summary = ref(null);
    const keyFreq = ref([]);
    const keyDurations = ref([]);
    const loading = ref(false);
    const error = ref("");

    const buildQuery = () => {
      const params = new URLSearchParams();
      if (filters.app) params.set("app", filters.app);
      if (filters.window) params.set("window", filters.window);
      return params.toString();
    };

    const reload = async () => {
      loading.value = true;
      error.value = "";
      try {
        const query = buildQuery();
        const [sum, freq, durations] = await Promise.all([
          fetchJson("/api/summary", query),
          fetchJson("/api/key-frequency", query),
          fetchJson("/api/key-durations", query),
        ]);
        summary.value = sum;
        keyFreq.value = freq.keys || [];
        keyDurations.value = durations.keys || [];

        await nextTick();
        const freqSlice = keyFreq.value.slice(0, 30);
        renderBar("keyFreq", {
          labels: freqSlice.map((k) => k.key),
          data: freqSlice.map((k) => k.count),
          label: "Keystrokes",
          color: "#2a84ff",
        });

        const durSlice = keyDurations.value.slice(0, 30);
        renderBar("duration", {
          labels: durSlice.map((k) => k.key),
          data: durSlice.map((k) => Number(k.avgDuration)),
          label: "Avg Duration (s)",
          color: "#6dd3a0",
        });
      } catch (err) {
        console.error(err);
        error.value = err.message || "Failed to load data";
      } finally {
        loading.value = false;
      }
    };

    onMounted(() => reload());

    return {
      filters,
      summary,
      keyFreq,
      keyDurations,
      loading,
      error,
      reload,
    };
  },
}).mount("#app");
