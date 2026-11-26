package protocol

// Control messages are JSON text frames; every message has a "type" field.
// Adding fields is backward compatible: receivers ignore unknown fields.

const (
	TypeHello      = "hello"       // worker -> coordinator
	TypeWelcome    = "welcome"     // coordinator -> worker
	TypeHeartbeat  = "heartbeat"   // worker -> coordinator
	TypeAssign     = "assign"      // coordinator -> worker
	TypeGetWeights = "get_weights" // worker -> coordinator
	TypeRevoke     = "revoke"      // coordinator -> worker
	TypeError      = "error"       // either direction
)

type Envelope struct {
	Type string `json:"type"`
}

type Hello struct {
	Type    string `json:"type"`
	Backend string `json:"backend"` // "webgpu" or "webgl"
	Label   string `json:"label"`
}

type TrainConfig struct {
	BatchSize    int     `json:"batchSize"`
	LearningRate float64 `json:"learningRate"`
}

type Welcome struct {
	Type        string      `json:"type"`
	WorkerID    string      `json:"workerId"`
	HeartbeatMs int         `json:"heartbeatMs"`
	Config      TrainConfig `json:"config"`
}

type Heartbeat struct {
	Type       string `json:"type"`
	NumTensors int    `json:"numTensors"`
	Busy       bool   `json:"busy"`
}

type Assign struct {
	Type        string `json:"type"`
	LeaseID     string `json:"leaseId"`
	Shard       int    `json:"shard"`
	BaseVersion int    `json:"baseVersion"`
	ExpiresAt   int64  `json:"expiresAt"` // Unix milliseconds
}

type GetWeights struct {
	Type    string `json:"type"`
	Version int    `json:"version"`
}

type Revoke struct {
	Type    string `json:"type"`
	LeaseID string `json:"leaseId"`
}

type Error struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}
