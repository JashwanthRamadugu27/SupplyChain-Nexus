import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Database,
  GitBranch,
  Globe2,
  KeyRound,
  LayoutDashboard,
  Loader2,
  Lock,
  LogIn,
  Network,
  RefreshCw,
  Route,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  Truck,
  Users,
  X,
  Zap,
} from "lucide-react";
import "./index.css";

const API = "http://127.0.0.1:8000";

type Page =
  | "dashboard"
  | "network"
  | "disruption"
  | "prediction"
  | "recovery"
  | "decision"
  | "analytics";

interface Hub {
  hub?: string;
  logistic_hub?: string;
  orders?: number;
  units?: number;
  late_rate?: number;
  [key: string]: any;
}

interface NetworkData {
  nodes?: number;
  connections?: number;
  ports?: number;
  hubs?: number;
  customers?: number;
  [key: string]: any;
}

function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [page, setPage] = useState<Page>("dashboard");
  const [apiOnline, setApiOnline] = useState(false);
  const [network, setNetwork] = useState<NetworkData>({});
  const [hubs, setHubs] = useState<Hub[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem("supplychain_logged_in");

    if (saved === "true") {
      setLoggedIn(true);
    }

    loadSystemData();
  }, []);

  async function loadSystemData() {
    try {
      const health = await fetch(`${API}/health`);
      setApiOnline(health.ok);

      const networkResponse = await fetch(`${API}/network`);
      if (networkResponse.ok) {
        const data = await networkResponse.json();
        setNetwork(data);
      }

      const hubsResponse = await fetch(`${API}/hubs`);
      if (hubsResponse.ok) {
        const data = await hubsResponse.json();

        if (Array.isArray(data)) {
          setHubs(data);
        } else if (Array.isArray(data.hubs)) {
          setHubs(data.hubs);
        } else if (Array.isArray(data.data)) {
          setHubs(data.data);
        }
      }
    } catch {
      setApiOnline(false);
    }
  }

  function login() {
    localStorage.setItem("supplychain_logged_in", "true");
    setLoggedIn(true);
  }

  function logout() {
    localStorage.removeItem("supplychain_logged_in");
    setLoggedIn(false);
  }

  if (!loggedIn) {
    return <LoginPage onLogin={login} />;
  }

  return (
    <div className="app-shell">
      <Sidebar page={page} setPage={setPage} logout={logout} />

      <main className="main-area">
        <Topbar
          page={page}
          apiOnline={apiOnline}
          onRefresh={loadSystemData}
        />

        <div className="page-container">
          {page === "dashboard" && (
            <Dashboard
              network={network}
              hubs={hubs}
              setPage={setPage}
            />
          )}

          {page === "network" && (
            <NetworkPage network={network} hubs={hubs} />
          )}

          {page === "disruption" && (
            <DisruptionPage hubs={hubs} />
          )}

          {page === "prediction" && (
            <PredictionPage />
          )}

          {page === "recovery" && (
            <RecoveryPage />
          )}

          {page === "decision" && (
            <DecisionPage />
          )}

          {page === "analytics" && (
            <AnalyticsPage />
          )}
        </div>
      </main>
    </div>
  );
}

/* =========================
   LOGIN
========================= */

function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="login-screen">
      <div className="login-orb orb-one" />
      <div className="login-orb orb-two" />

      <div className="login-left">
        <div className="brand-large">
          <div className="brand-icon-large">
            <Network size={25} />
          </div>
          <span>SupplyChain Nexus</span>
        </div>

        <div className="login-copy">
          <div className="eyebrow">
            <Sparkles size={14} />
            DISRUPTION INTELLIGENCE
          </div>

          <h1>
            Understand disruption.
            <br />
            <span>Plan recovery.</span>
          </h1>

          <p>
            A data and ML-driven decision-support system for studying
            supply-chain disruption propagation, impact, and recovery options.
          </p>

          <div className="login-features">
            <Feature icon={<Network size={18} />} text="Network-aware propagation analysis" />
            <Feature icon={<BrainCircuit size={18} />} text="ML-based risk prediction" />
            <Feature icon={<Route size={18} />} text="Recovery option evaluation" />
          </div>
        </div>
      </div>

      <div className="login-card">
        <div className="login-card-header">
          <div className="mini-icon">
            <Lock size={18} />
          </div>

          <div>
            <h2>System access</h2>
            <p>Sign in to SupplyChain Nexus</p>
          </div>
        </div>

        <label>Email</label>
        <div className="input-wrap">
          <Users size={17} />
          <input
            type="email"
            placeholder="manager@organization.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        <label>Password</label>
        <div className="input-wrap">
          <KeyRound size={17} />
          <input
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        <button className="login-button" onClick={onLogin}>
          <LogIn size={18} />
          Enter Command Center
        </button>

        <div className="demo-note">
          <ShieldCheck size={15} />
          Demo authentication for the current prototype
        </div>
      </div>
    </div>
  );
}

function Feature({
  icon,
  text,
}: {
  icon: React.ReactNode;
  text: string;
}) {
  return (
    <div className="login-feature">
      <div>{icon}</div>
      <span>{text}</span>
    </div>
  );
}

/* =========================
   SIDEBAR
========================= */

function Sidebar({
  page,
  setPage,
  logout,
}: {
  page: Page;
  setPage: (page: Page) => void;
  logout: () => void;
}) {
  const items: {
    id: Page;
    label: string;
    icon: React.ReactNode;
  }[] = [
    {
      id: "dashboard",
      label: "Command Center",
      icon: <LayoutDashboard size={18} />,
    },
    {
      id: "network",
      label: "Supply Network",
      icon: <Network size={18} />,
    },
    {
      id: "disruption",
      label: "Disruption Analysis",
      icon: <AlertTriangle size={18} />,
    },
    {
      id: "prediction",
      label: "ML Predictions",
      icon: <BrainCircuit size={18} />,
    },
    {
      id: "recovery",
      label: "Recovery Planning",
      icon: <Route size={18} />,
    },
    {
      id: "decision",
      label: "Decision Support",
      icon: <Target size={18} />,
    },
    {
      id: "analytics",
      label: "Analytics",
      icon: <BarChart3 size={18} />,
    },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">
          <Network size={20} />
        </div>

        <div>
          <strong>SupplyChain</strong>
          <span>NEXUS</span>
        </div>
      </div>

      <div className="sidebar-label">INTELLIGENCE</div>

      <nav>
        {items.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${page === item.id ? "active" : ""}`}
            onClick={() => setPage(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
            {page === item.id && <ChevronRight size={15} />}
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="system-status">
          <div className="status-dot" />
          <div>
            <strong>Decision engine</strong>
            <span>Operational</span>
          </div>
        </div>

        <button className="logout-button" onClick={logout}>
          <LogIn size={16} />
          Sign out
        </button>
      </div>
    </aside>
  );
}

/* =========================
   TOPBAR
========================= */

function Topbar({
  page,
  apiOnline,
  onRefresh,
}: {
  page: Page;
  apiOnline: boolean;
  onRefresh: () => void;
}) {
  const titles: Record<Page, string> = {
    dashboard: "Command Center",
    network: "Supply Network",
    disruption: "Disruption Analysis",
    prediction: "ML Predictions",
    recovery: "Recovery Planning",
    decision: "Decision Support",
    analytics: "Analytics",
  };

  return (
    <header className="topbar">
      <div>
        <span className="breadcrumb">SUPPLYCHAIN NEXUS /</span>
        <h2>{titles[page]}</h2>
      </div>

      <div className="topbar-actions">
        <div className="api-status">
          <span className={apiOnline ? "online-dot" : "offline-dot"} />
          {apiOnline ? "API Connected" : "API Offline"}
        </div>

        <button className="icon-button" onClick={onRefresh}>
          <RefreshCw size={17} />
        </button>

        <div className="profile">
          <div className="profile-avatar">M</div>
          <div>
            <strong>Supply Manager</strong>
            <span>Decision Support</span>
          </div>
        </div>
      </div>
    </header>
  );
}

/* =========================
   DASHBOARD
========================= */

function Dashboard({
  network,
  hubs,
  setPage,
}: {
  network: NetworkData;
  hubs: Hub[];
  setPage: (page: Page) => void;
}) {
  const nodeCount = network.nodes ?? 42;
  const connections = network.connections ?? 343;
  const ports = network.ports ?? 5;
  const customers = network.customers ?? 28;

  return (
    <div className="fade-page">
      <section className="hero">
        <div className="hero-glow" />

        <div className="hero-content">
          <div className="eyebrow">
            <Activity size={14} />
            SUPPLY CHAIN INTELLIGENCE
          </div>

          <h1>
            See the disruption
            <br />
            <span>before it spreads.</span>
          </h1>

          <p>
            Analyze connected supply-chain networks, understand disruption
            propagation, predict risk, and evaluate recovery options.
          </p>

          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={() => setPage("disruption")}
            >
              Analyze a Disruption
              <ArrowRight size={17} />
            </button>

            <button
              className="secondary-button"
              onClick={() => setPage("network")}
            >
              Explore Network
            </button>
          </div>
        </div>

        <NetworkPreview />
      </section>

      <div className="section-title">
        <div>
          <span>LIVE SYSTEM VIEW</span>
          <h2>Supply-chain intelligence</h2>
        </div>
      </div>

      <div className="metric-grid">
        <MetricCard
          icon={<Network size={20} />}
          label="Network Nodes"
          value={nodeCount}
          suffix="nodes"
        />

        <MetricCard
          icon={<GitBranch size={20} />}
          label="Connections"
          value={connections}
          suffix="links"
        />

        <MetricCard
          icon={<Globe2 size={20} />}
          label="Origin Ports"
          value={ports}
          suffix="ports"
        />

        <MetricCard
          icon={<Users size={20} />}
          label="Customers"
          value={customers}
          suffix="nodes"
        />
      </div>

      <div className="dashboard-grid">
        <section className="panel large-panel">
          <div className="panel-header">
            <div>
              <span>NETWORK INTELLIGENCE</span>
              <h3>Disruption propagation map</h3>
            </div>

            <button
              className="small-button"
              onClick={() => setPage("network")}
            >
              Open network
              <ArrowRight size={14} />
            </button>
          </div>

          <LargeNetworkVisual />
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <span>RECOVERY READINESS</span>
              <h3>Network redundancy</h3>
            </div>
          </div>

          <div className="readiness">
            <div className="readiness-ring">
              <span>8</span>
              <small>alternatives</small>
            </div>

            <div>
              <strong>Alternative paths detected</strong>
              <p>
                Connected customers have multiple potential hub alternatives
                in the current network structure.
              </p>
            </div>
          </div>

          <div className="mini-list">
            {hubs.slice(0, 5).map((hub, index) => {
              const name =
                hub.logistic_hub || hub.hub || `Hub ${index + 1}`;

              return (
                <div className="mini-row" key={name}>
                  <CircleDot size={15} />
                  <span>{name}</span>
                  <b>{index + 1}</b>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  suffix,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  suffix: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
      <small>{suffix}</small>
    </div>
  );
}

/* =========================
   NETWORK
========================= */

function NetworkPage({
  network,
  hubs,
}: {
  network: NetworkData;
  hubs: Hub[];
}) {
  return (
    <div className="fade-page">
      <PageIntro
        eyebrow="NETWORK TOPOLOGY"
        title="Supply-chain network"
        description="Explore the connected structure of ports, logistics hubs, and customers used for disruption propagation analysis."
      />

      <div className="network-stat-row">
        <StatBox label="Nodes" value={network.nodes ?? 42} />
        <StatBox label="Connections" value={network.connections ?? 343} />
        <StatBox label="Ports" value={network.ports ?? 5} />
        <StatBox label="Hubs" value={network.hubs ?? 9} />
        <StatBox label="Customers" value={network.customers ?? 28} />
      </div>

      <section className="panel network-panel">
        <div className="panel-header">
          <div>
            <span>INTERACTIVE TOPOLOGY</span>
            <h3>Port → Hub → Customer relationships</h3>
          </div>

          <div className="network-legend">
            <span><i className="legend-port" /> Port</span>
            <span><i className="legend-hub" /> Hub</span>
            <span><i className="legend-customer" /> Customer</span>
          </div>
        </div>

        <LargeNetworkVisual />
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <span>LOGISTICS HUBS</span>
            <h3>Network nodes</h3>
          </div>
        </div>

        <div className="hub-grid">
          {hubs.map((hub, index) => {
            const name =
              hub.logistic_hub ||
              hub.hub ||
              `Logistics Hub ${index + 1}`;

            return (
              <div className="hub-card" key={`${name}-${index}`}>
                <div className="hub-card-icon">
                  <Network size={17} />
                </div>

                <div>
                  <strong>{name}</strong>
                  <span>Supply-chain node</span>
                </div>

                <ChevronRight size={16} />
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

/* =========================
   DISRUPTION
========================= */

function DisruptionPage({ hubs }: { hubs: Hub[] }) {
  const [selectedHub, setSelectedHub] = useState("Venlo");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  async function analyze() {
    setLoading(true);
    setResult(null);
    setError("");

    try {
      console.log("Sending disruption request:", selectedHub);
      console.log("API URL:", `${API}/disruption/analyze`);

      const response = await fetch(`${API}/disruption/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          hub: selectedHub,
        }),
      });

      console.log("Response status:", response.status);

      const responseText = await response.text();

      console.log("Response:", responseText);

      if (!response.ok) {
        setError(
          `Backend error (${response.status}): ${responseText}`
        );
        return;
      }

      const data = JSON.parse(responseText);

      console.log("Disruption analysis result:", data);

      setResult(data);
    } catch (err: any) {
      console.error("Disruption analysis failed:", err);

      setError(
        `Connection error: ${
          err?.message || "Could not connect to the FastAPI backend."
        }`
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fade-page">

      <PageIntro
        eyebrow="DISRUPTION INTELLIGENCE"
        title="Propagation analysis"
        description="Simulate a disruption at a supply-chain node and analyze how the event can propagate through connected parts of the network."
      />

      {/* =========================
          DISRUPTION CONTROL
      ========================= */}

      <section className="panel disruption-control">

        <div>
          <span>DISRUPTION SCENARIO</span>

          <h3>Select a disrupted logistics hub</h3>

          <p>
            The system will analyze the connected network and calculate
            the potential downstream impact.
          </p>
        </div>

        <div className="scenario-controls">

          <div className="select-wrap">

            <Network size={17} />

            <select
              value={selectedHub}
              onChange={(e) => {
                setSelectedHub(e.target.value);
                setResult(null);
                setError("");
              }}
            >

              {hubs.length > 0 ? (

                hubs.map((hub, index) => {

                  const name =
                    hub.logistic_hub ||
                    hub.hub ||
                    `Hub ${index + 1}`;

                  return (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  );
                })

              ) : (

                <>
                  <option value="Venlo">Venlo</option>
                  <option value="Hamburg">Hamburg</option>
                  <option value="Warsaw">Warsaw</option>
                  <option value="Dusseldorf">Dusseldorf</option>
                  <option value="Rome">Rome</option>
                  <option value="Lille">Lille</option>
                  <option value="Zaragoza">Zaragoza</option>
                  <option value="Liege">Liege</option>
                  <option value="Bratislava">Bratislava</option>
                </>

              )}

            </select>

          </div>

          <button
            className="primary-button"
            onClick={analyze}
            disabled={loading}
          >

            {loading ? "Analyzing..." : "Analyze Disruption"}

            {!loading && <ArrowRight size={17} />}

          </button>

        </div>

      </section>


      {/* =========================
          ERROR MESSAGE
      ========================= */}

      {error && (

        <section
          className="panel"
          style={{
            marginTop: 16,
            border: "1px solid rgba(239,68,68,0.45)",
            background: "rgba(239,68,68,0.08)",
          }}
        >

          <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>

            <AlertTriangle
              size={22}
              style={{ color: "#ef4444", marginTop: 2 }}
            />

            <div>

              <h3 style={{ margin: 0 }}>
                Analysis failed
              </h3>

              <p style={{ marginTop: 8, marginBottom: 0 }}>
                {error}
              </p>

            </div>

          </div>

        </section>

      )}


      {/* =========================
          RESULT
      ========================= */}

      {result && (

        <div style={{ marginTop: 20 }}>

          <div className="result-grid">

            <ResultCard
              label="Disrupted Node"
              value={result.disrupted_hub ?? selectedHub}
              icon={<AlertTriangle size={19} />}
            />

            <ResultCard
              label="Orders Affected"
              value={formatNumber(
                result.affected_orders ??
                result.orders_affected ??
                result.orders ??
                0
              )}
              icon={<Database size={19} />}
            />

            <ResultCard
              label="Units Affected"
              value={formatNumber(
                result.affected_units ??
                result.units_affected ??
                result.units ??
                0
              )}
              icon={<Truck size={19} />}
            />

            <ResultCard
              label="Customers Affected"
              value={
                result.affected_customers ??
                result.customers_affected ??
                0
              }
              icon={<Users size={19} />}
            />

            <ResultCard
              label="Historical Late Rate"
              value={`${(
                result.historical_late_rate_percentage ??
                ((result.historical_late_rate ?? 0) * 100)
              ).toFixed(2)}%`}
              icon={<Activity size={19} />}
            />

          </div>


          {/* =========================
              BACKEND MESSAGE
          ========================= */}

          <section className="panel" style={{ marginTop: 18 }}>

            <div className="panel-header">

              <div>

                <span>ANALYSIS RESULT</span>

                <h3>
                  Disruption impact detected
                </h3>

              </div>

              <div
                style={{
                  padding: "6px 10px",
                  borderRadius: 999,
                  background: "rgba(34,197,94,0.12)",
                  color: "#22c55e",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                ANALYSIS COMPLETE
              </div>

            </div>

            <p style={{ marginTop: 10 }}>
              {result.message ??
                `The disruption at ${selectedHub} has been analyzed using the supply-chain network data.`}
            </p>

          </section>


          {/* =========================
              AFFECTED CUSTOMERS
          ========================= */}

          {Array.isArray(result.affected_customer_names) &&
            result.affected_customer_names.length > 0 && (

              <section className="panel" style={{ marginTop: 18 }}>

                <div className="panel-header">

                  <div>

                    <span>DOWNSTREAM IMPACT</span>

                    <h3>
                      Affected customers
                    </h3>

                  </div>

                </div>

                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 8,
                    marginTop: 14,
                  }}
                >

                  {result.affected_customer_names.map(
                    (customer: string) => (

                      <span
                        key={customer}
                        style={{
                          padding: "7px 11px",
                          borderRadius: 8,
                          background: "rgba(255,255,255,0.05)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          fontSize: 13,
                        }}
                      >
                        {customer}
                      </span>

                    )
                  )}

                </div>

              </section>

            )}


          {/* =========================
              ALTERNATIVE HUBS
          ========================= */}

          {result.alternative_hubs &&
            Object.keys(result.alternative_hubs).length > 0 && (

              <section className="panel" style={{ marginTop: 18 }}>

                <div className="panel-header">

                  <div>

                    <span>RECOVERY PATHS</span>

                    <h3>
                      Available alternative hubs
                    </h3>

                  </div>

                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: 12,
                    marginTop: 14,
                  }}
                >

                  {Object.entries(
                    result.alternative_hubs
                  )
                    .slice(0, 6)
                    .map(
                      ([customer, alternatives]: [
                        string,
                        any
                      ]) => (

                        <div
                          key={customer}
                          style={{
                            padding: 14,
                            borderRadius: 10,
                            background:
                              "rgba(255,255,255,0.035)",
                            border:
                              "1px solid rgba(255,255,255,0.08)",
                          }}
                        >

                          <strong
                            style={{
                              fontSize: 13,
                            }}
                          >
                            {customer}
                          </strong>

                          <p
                            style={{
                              marginTop: 8,
                              fontSize: 12,
                              opacity: 0.7,
                            }}
                          >
                            {Array.isArray(alternatives)
                              ? `${alternatives.length} alternative hubs available`
                              : "Alternative recovery paths available"}
                          </p>

                          {Array.isArray(alternatives) && (

                            <div
                              style={{
                                display: "flex",
                                flexWrap: "wrap",
                                gap: 6,
                              }}
                            >

                              {alternatives.map(
                                (hub: string) => (

                                  <span
                                    key={hub}
                                    style={{
                                      padding:
                                        "5px 8px",
                                      borderRadius: 6,
                                      background:
                                        "rgba(255,255,255,0.06)",
                                      fontSize: 11,
                                    }}
                                  >
                                    {hub}
                                  </span>

                                )
                              )}

                            </div>

                          )}

                        </div>

                      )
                    )}

                </div>

                <p
                  style={{
                    marginTop: 14,
                    opacity: 0.6,
                    fontSize: 12,
                  }}
                >
                  Showing recovery alternatives for the first
                  6 affected customers. Recovery Planning will
                  evaluate these alternatives in detail.
                </p>

              </section>

            )}

        </div>

      )}


      {/* =========================
          EMPTY STATE
      ========================= */}

      {!result && !loading && !error && (

        <section className="empty-analysis">

          <div className="empty-icon">
            <Zap size={24} />
          </div>

          <h3>
            Ready for disruption simulation
          </h3>

          <p>
            Select a network node and run the analysis to see
            how the disruption can propagate through connected
            supply-chain nodes.
          </p>

        </section>

      )}


      {/* =========================
          LOADING STATE
      ========================= */}

      {loading && (

        <section className="empty-analysis">

          <div className="empty-icon">
            <Loader2
              size={24}
              className="spin"
            />
          </div>

          <h3>
            Analyzing {selectedHub} disruption...
          </h3>

          <p>
            Connecting to the SupplyChain Nexus analysis engine
            and calculating downstream impact.
          </p>

        </section>

      )}


      {/* =========================
          NETWORK VISUAL
      ========================= */}

      <section className="panel">

        <div className="panel-header">

          <div>

            <span>PROPAGATION VIEW</span>

            <h3>
              Network response
            </h3>

          </div>

        </div>

        <LargeNetworkVisual
          disrupted={Boolean(result)}
          disruptedHub={
            result?.disrupted_hub ?? selectedHub
          }
        />

      </section>

    </div>
  );
}

/* =========================
   PREDICTION
========================= */

function PredictionPage() {
  const [units, setUnits] = useState("500");
  const [weight, setWeight] = useState("10");
  const [distance, setDistance] = useState("1000");

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);

  async function predict() {
    setLoading(true);

    try {
      const response = await fetch(`${API}/prediction`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          origin_port: "Rotterdam",
          three_pl: "v_002",
          customs_procedures: "DTD",
          logistic_hub: "Venlo",
          customer: "Customer_1",
          units: Number(units),
          weight: Number(weight),
          weight_class: 0,
          material_handling: "AMBIENT",
          distance: Number(distance),
          cost_per_unit: 1,
          co2_per_unit: 1,
          product_data_missing: 0,
          route_data_missing: 0,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data);
      }
    } catch {
      setPrediction(null);
    }

    setLoading(false);
  }

  return (
    <div className="fade-page">
      <PageIntro
        eyebrow="MACHINE LEARNING"
        title="Disruption risk prediction"
        description="Use the trained ML pipeline to estimate the probability of a shipment being late based on supply-chain characteristics."
      />

      <div className="prediction-layout">
        <section className="panel prediction-form">
          <div className="panel-header">
            <div>
              <span>INPUT PARAMETERS</span>
              <h3>Shipment scenario</h3>
            </div>
          </div>

          <div className="form-grid">
            <Field
              label="Units"
              value={units}
              onChange={setUnits}
            />

            <Field
              label="Weight"
              value={weight}
              onChange={setWeight}
            />

            <Field
              label="Distance (km)"
              value={distance}
              onChange={setDistance}
            />
          </div>

          <button className="primary-button full-width" onClick={predict}>
            {loading ? "Running ML model..." : "Run Risk Prediction"}
            {!loading && <BrainCircuit size={17} />}
          </button>
        </section>

        <section className="panel prediction-result">
          <div className="panel-header">
            <div>
              <span>MODEL OUTPUT</span>
              <h3>Predicted risk</h3>
            </div>
          </div>

          {prediction ? (
            <>
              <div className="risk-number">
                {Number(
                  prediction.late_probability ??
                    prediction.probability ??
                    0
                ).toFixed(1)}
                <span>%</span>
              </div>

              <div className="risk-label">
                {prediction.risk_level ?? "Predicted Risk"}
              </div>

              <div className="risk-bar">
                <div
                  style={{
                    width: `${Math.min(
                      100,
                      Number(
                        prediction.late_probability ??
                          prediction.probability ??
                          0
                      )
                    )}%`,
                  }}
                />
              </div>

              <p>
                The prediction is generated by the trained SupplyChain Nexus
                ML pipeline.
              </p>
            </>
          ) : (
            <div className="prediction-empty">
              <BrainCircuit size={30} />
              <strong>No prediction yet</strong>
              <span>
                Enter scenario values and run the ML model.
              </span>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

/* =========================
   RECOVERY
========================= */

function RecoveryPage() {
  const [selectedHub, setSelectedHub] = useState("Lille");
  const [selectedCustomer, setSelectedCustomer] = useState("Amsterdam");
  const [options, setOptions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const hubs = [
    "Bratislava",
    "Dusseldorf",
    "Hamburg",
    "Liege",
    "Lille",
    "Rome",
    "Venlo",
    "Warsaw",
    "Zaragoza",
  ];

  const customers = [
    "Amsterdam",
    "Athens",
    "Barcelona",
    "Berlin",
    "Bordeaux",
    "Bremen",
    "Bucharest",
    "Budapest",
    "Cologne",
    "Copenhagen",
    "Hanover",
    "Helsinki",
    "Lisbon",
    "Lyon",
    "Madrid",
    "Malmö",
    "Marseille",
    "Milan",
    "Munich",
    "Naples",
    "Paris",
    "Porto",
    "Prague",
    "Rome",
    "Stockholm",
    "Turin",
    "Valencia",
    "Vienna",
  ];

  async function evaluateRecovery() {
    setLoading(true);
    setError("");
    setOptions([]);

    try {
      const response = await fetch(`${API}/recovery/options`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          hub: selectedHub,
          customer: selectedCustomer,
        }),
      });

      if (!response.ok) {
        throw new Error("Recovery analysis failed.");
      }

      const data = await response.json();

      setOptions(data.options || []);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the recovery API."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fade-page">
      <PageIntro
        eyebrow="RECOVERY PLANNING"
        title="Evaluate recovery options"
        description="Compare alternative supply-chain paths after a disruption using cost, distance, reliability, CO₂ and route coverage."
      />

      <section className="recovery-banner">
        <div className="recovery-icon">
          <RefreshCw size={21} />
        </div>

        <div>
          <span>RECOVERY SCENARIO</span>
          <strong>
            {selectedHub} disruption → {selectedCustomer}
          </strong>

          <p>
            The system evaluates alternative hubs and calculates a recovery
            score for each available path.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <span>SCENARIO INPUT</span>
            <h3>Select disruption and customer</h3>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr auto",
            gap: "16px",
            alignItems: "end",
            marginBottom: "24px",
          }}
        >
          <div>
            <label
              style={{
                display: "block",
                fontSize: "12px",
                marginBottom: "8px",
                opacity: 0.7,
              }}
            >
              Disrupted logistics hub
            </label>

            <select
              value={selectedHub}
              onChange={(e) => setSelectedHub(e.target.value)}
              style={{
                width: "100%",
                padding: "12px",
                borderRadius: "8px",
                border: "1px solid rgba(255,255,255,0.12)",
                background: "#111827",
                color: "white",
              }}
            >
              {hubs.map((hub) => (
                <option key={hub} value={hub}>
                  {hub}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label
              style={{
                display: "block",
                fontSize: "12px",
                marginBottom: "8px",
                opacity: 0.7,
              }}
            >
              Downstream customer
            </label>

            <select
              value={selectedCustomer}
              onChange={(e) => setSelectedCustomer(e.target.value)}
              style={{
                width: "100%",
                padding: "12px",
                borderRadius: "8px",
                border: "1px solid rgba(255,255,255,0.12)",
                background: "#111827",
                color: "white",
              }}
            >
              {customers.map((customer) => (
                <option key={customer} value={customer}>
                  {customer}
                </option>
              ))}
            </select>
          </div>

          <button
            className="evaluate-button"
            onClick={evaluateRecovery}
            disabled={loading}
            style={{
              height: "44px",
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Evaluating..." : "Evaluate"}
            <ArrowRight size={14} />
          </button>
        </div>

        {error && (
          <div
            style={{
              padding: "14px",
              borderRadius: "8px",
              marginBottom: "20px",
              background: "rgba(239,68,68,0.1)",
              border: "1px solid rgba(239,68,68,0.25)",
              color: "#fca5a5",
            }}
          >
            {error}
          </div>
        )}

        {options.length > 0 && (
          <>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "16px",
              }}
            >
              <div>
                <span
                  style={{
                    fontSize: "11px",
                    opacity: 0.6,
                  }}
                >
                  OPTION EVALUATION
                </span>

                <h3 style={{ marginTop: "5px" }}>
                  {options.length} recovery paths evaluated
                </h3>
              </div>

              <span
                style={{
                  fontSize: "12px",
                  opacity: 0.65,
                }}
              >
                Ranked by recovery score
              </span>
            </div>

            <div className="recovery-table">
              <div className="table-head">
                <span>Alternative hub</span>
                <span>Distance</span>
                <span>Late rate</span>
                <span>Coverage</span>
                <span>Recovery score</span>
                <span>Rank</span>
              </div>

              {options
                .slice()
                .sort(
                  (a, b) =>
                    Number(a.recovery_rank) -
                    Number(b.recovery_rank)
                )
                .map((option) => (
                  <div
                    className="table-row"
                    key={option.alternative_hub}
                  >
                    <strong>
                      <Network size={16} />
                      {option.alternative_hub}
                    </strong>

                    <span>
                      {Number(option.alternative_distance).toFixed(1)} km
                    </span>

                    <span>
                      {(Number(
                        option.alternative_historical_late_rate
                      ) * 100).toFixed(1)}
                      %
                    </span>

                    <span>
                      {(Number(
                        option.route_unit_coverage
                      ) * 100).toFixed(1)}
                      %
                    </span>

                    <span
                      style={{
                        fontWeight: 700,
                      }}
                    >
                      {Number(option.recovery_score).toFixed(2)}
                    </span>

                    <span className="risk-pill">
                      #{option.recovery_rank}
                    </span>
                  </div>
                ))}
            </div>
          </>
        )}

        {!loading && options.length === 0 && !error && (
          <div
            style={{
              padding: "32px",
              textAlign: "center",
              opacity: 0.6,
            }}
          >
            Select a disruption scenario and click{" "}
            <strong>Evaluate</strong> to calculate recovery paths.
          </div>
        )}
      </section>
    </div>
  );
}

/* =========================
   DECISION SUPPORT
========================= */

function DecisionPage() {
  return (
    <div className="fade-page">
      <PageIntro
        eyebrow="DECISION SUPPORT"
        title="Recovery decision workspace"
        description="Bring disruption impact, ML risk, and recovery-option evaluation together so decision-makers can compare the available choices."
      />

      <div className="decision-flow">
        <DecisionStep
          number="01"
          title="Disruption"
          text="Identify the affected network node."
          icon={<AlertTriangle size={20} />}
        />

        <ArrowRight className="flow-arrow" />

        <DecisionStep
          number="02"
          title="Propagation"
          text="Analyze connected downstream impact."
          icon={<Network size={20} />}
        />

        <ArrowRight className="flow-arrow" />

        <DecisionStep
          number="03"
          title="ML Risk"
          text="Estimate shipment delay risk."
          icon={<BrainCircuit size={20} />}
        />

        <ArrowRight className="flow-arrow" />

        <DecisionStep
          number="04"
          title="Recovery"
          text="Compare alternative paths."
          icon={<Route size={20} />}
        />
      </div>

      <section className="panel decision-panel">
        <div className="decision-header">
          <div>
            <span>DECISION WORKSPACE</span>
            <h3>Recovery option comparison</h3>
          </div>

          <div className="decision-status">
            <CheckCircle2 size={16} />
            Analysis ready
          </div>
        </div>

        <div className="comparison-grid">
          <ComparisonCard
            title="Option A"
            hub="Hamburg"
            score="0.82"
            description="Balanced recovery option across cost, distance, risk and emissions."
          />

          <ComparisonCard
            title="Option B"
            hub="Dusseldorf"
            score="0.78"
            description="Shorter network distance with different operational trade-offs."
          />

          <ComparisonCard
            title="Option C"
            hub="Liege"
            score="0.71"
            description="Alternative route providing additional network redundancy."
          />
        </div>

        <div className="decision-note">
          <ShieldCheck size={18} />
          <div>
            <strong>Human decision remains final</strong>
            <p>
              SupplyChain Nexus provides analysis and option evaluation. The
              organization decides which recovery action is appropriate.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

function DecisionStep({
  number,
  title,
  text,
  icon,
}: {
  number: string;
  title: string;
  text: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="decision-step">
      <div className="decision-number">{number}</div>
      <div className="decision-icon">{icon}</div>
      <strong>{title}</strong>
      <p>{text}</p>
    </div>
  );
}

function ComparisonCard({
  title,
  hub,
  score,
  description,
}: {
  title: string;
  hub: string;
  score: string;
  description: string;
}) {
  return (
    <div className="comparison-card">
      <span>{title}</span>
      <h4>{hub}</h4>

      <div className="comparison-score">
        {score}
        <small>evaluation</small>
      </div>

      <p>{description}</p>

      <button>
        View details
        <ArrowRight size={14} />
      </button>
    </div>
  );
}

/* =========================
   ANALYTICS
========================= */

function AnalyticsPage() {
  return (
    <div className="fade-page">
      <PageIntro
        eyebrow="SYSTEM ANALYTICS"
        title="Supply-chain analytics"
        description="Historical and model-level indicators supporting disruption analysis and recovery planning."
      />

      <div className="analytics-grid">
        <AnalyticsCard
          icon={<TrendingUp size={20} />}
          title="ML ROC-AUC"
          value="0.8035"
          subtitle="Logistic regression"
        />

        <AnalyticsCard
          icon={<BrainCircuit size={20} />}
          title="ML F1 Score"
          value="0.5661"
          subtitle="Threshold 0.25"
        />

        <AnalyticsCard
          icon={<Network size={20} />}
          title="Network Nodes"
          value="42"
          subtitle="Ports + hubs + customers"
        />

        <AnalyticsCard
          icon={<Route size={20} />}
          title="Recovery Options"
          value="2,016"
          subtitle="Evaluated combinations"
        />
      </div>

      <section className="panel">
        <div className="panel-header">
          <div>
            <span>PROJECT ANALYTICS</span>
            <h3>Decision-support pipeline</h3>
          </div>
        </div>

        <div className="pipeline">
          <PipelineItem
            number="01"
            title="Data"
            text="Historical supply-chain records"
          />

          <PipelineItem
            number="02"
            title="Network"
            text="Connected supply-chain topology"
          />

          <PipelineItem
            number="03"
            title="Propagation"
            text="Disruption impact analysis"
          />

          <PipelineItem
            number="04"
            title="ML"
            text="Risk prediction and explanation"
          />

          <PipelineItem
            number="05"
            title="Recovery"
            text="Alternative option evaluation"
          />

          <PipelineItem
            number="06"
            title="Decision"
            text="Human decision support"
          />
        </div>
      </section>
    </div>
  );
}

/* =========================
   VISUAL COMPONENTS
========================= */

function NetworkPreview() {
  return (
    <div className="hero-network">
      <div className="network-line line-a" />
      <div className="network-line line-b" />
      <div className="network-line line-c" />
      <div className="network-line line-d" />
      <div className="network-line line-e" />

      <div className="hero-node port-node">P</div>
      <div className="hero-node hub-node">H</div>
      <div className="hero-node hub-node hub-two">H</div>

      <div className="hero-node customer-node c-one">C</div>
      <div className="hero-node customer-node c-two">C</div>
      <div className="hero-node customer-node c-three">C</div>
      <div className="hero-node customer-node c-four">C</div>

      <div className="network-tag">
        <Activity size={14} />
        LIVE NETWORK MODEL
      </div>
    </div>
  );
}

function LargeNetworkVisual({ disrupted = false, disruptedHub }: { disrupted?: boolean; disruptedHub?: string }) {
  return (
    <div className={`large-network ${disrupted ? "network-disrupted" : ""}`}>
      <div className="grid-lines" />

      <div className="connection l1" />
      <div className="connection l2" />
      <div className="connection l3" />
      <div className="connection l4" />
      <div className="connection l5" />
      <div className="connection l6" />
      <div className="connection l7" />
      <div className="connection l8" />
      <div className="connection l9" />
      <div className="connection l10" />
      <div className="connection l11" />

      <NetworkNode type="port" label="Rotterdam" x="8%" y="50%" />
      <NetworkNode type="port" label="Barcelona" x="8%" y="78%" />
      <NetworkNode type="hub" label="Venlo" x="38%" y="24%" alert={disruptedHub === "Venlo"} />
      <NetworkNode type="hub" label="Hamburg" x="38%" y="49%" alert={disruptedHub === "Hamburg"} />
      <NetworkNode type="hub" label="Warsaw" x="38%" y="76%" alert={disruptedHub === "Warsaw"} />

      <NetworkNode type="customer" label="Customer A" x="76%" y="17%" />
      <NetworkNode type="customer" label="Customer B" x="76%" y="34%" />
      <NetworkNode type="customer" label="Customer C" x="76%" y="51%" />
      <NetworkNode type="customer" label="Customer D" x="76%" y="68%" />
      <NetworkNode type="customer" label="Customer E" x="76%" y="85%" />

      {disrupted && (
        <div className="disruption-alert">
          <AlertTriangle size={14} />
          DISRUPTION PROPAGATION ACTIVE
        </div>
      )}

      <div className="network-label left-label">ORIGIN</div>
      <div className="network-label center-label">LOGISTICS HUB</div>
      <div className="network-label right-label">DOWNSTREAM</div>
    </div>
  );
}

function NetworkNode({
  type,
  label,
  x,
  y,
  alert = false,
}: {
  type: "port" | "hub" | "customer";
  label: string;
  x: string;
  y: string;
  alert?: boolean;
}) {
  return (
    <div
      className={`network-node ${type} ${alert ? "alert-node" : ""}`}
      style={{ left: x, top: y }}
    >
      <div className="node-core">
        {type === "port" && <Globe2 size={15} />}
        {type === "hub" && <Network size={15} />}
        {type === "customer" && <Users size={14} />}
      </div>

      <span>{label}</span>
    </div>
  );
}

function PageIntro({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="page-intro">
      <div className="eyebrow">
        <Sparkles size={14} />
        {eyebrow}
      </div>

      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}

function StatBox({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="stat-box">
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
    </div>
  );
}

function ResultCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
}) {
  return (
    <div className="result-card">
      <div className="result-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="input-wrap">
        <Search size={16} />
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    </div>
  );
}

function AnalyticsCard({
  icon,
  title,
  value,
  subtitle,
}: {
  icon: React.ReactNode;
  title: string;
  value: string;
  subtitle: string;
}) {
  return (
    <div className="analytics-card">
      <div className="analytics-icon">{icon}</div>
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{subtitle}</small>
    </div>
  );
}

function PipelineItem({
  number,
  title,
  text,
}: {
  number: string;
  title: string;
  text: string;
}) {
  return (
    <div className="pipeline-item">
      <div className="pipeline-number">{number}</div>
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

function formatNumber(value: number) {
  return Number(value || 0).toLocaleString();
}

export default App;