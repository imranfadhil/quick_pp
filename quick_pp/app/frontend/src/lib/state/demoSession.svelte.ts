import { untrack } from 'svelte';
import {
	API_BASE,
	authHeaders,
	getDemoSessionId,
	registerDemoClientHooks
} from '$lib/api/client';
import { appStatus } from '$lib/state/status.svelte';

export interface DemoStatusPayload {
	demo_mode: boolean;
	online_users: number;
	max_users: number;
	at_capacity: boolean;
	admitted?: boolean;
	session_active?: boolean;
	lag_warning: boolean;
	lag_warning_threshold: number;
	active_compute_jobs: number;
	max_compute_jobs: number;
	idle_timeout_seconds: number;
	hidden_idle_timeout_seconds: number;
	session_ttl_seconds: number;
}

const HEARTBEAT_INTERVAL_MS = 25_000;
const CAPACITY_RETRY_INTERVAL_MS = 10_000;
const IDLE_CHECK_INTERVAL_MS = 5_000;
const MIN_HEARTBEAT_GAP_MS = 5_000;

class DemoSessionManager {
	onlineUsers = $state<number>(1);
	maxUsers = $state<number>(6);
	lagWarning = $state<boolean>(false);
	lagWarningThreshold = $state<number>(3);
	activeComputeJobs = $state<number>(0);
	maxComputeJobs = $state<number>(2);
	atCapacity = $state<boolean>(false);
	isIdlePaused = $state<boolean>(false);
	idleReason = $state<'inactivity' | 'background'>('inactivity');
	idleTimeoutSeconds = $state<number>(600);
	hiddenIdleTimeoutSeconds = $state<number>(300);
	isReconnecting = $state<boolean>(false);

	// Plain non-reactive lifecycle fields (must NOT be $state to avoid $effect subscribe loops)
	private started = false;
	private lastActivityMs = Date.now();
	private lastHeartbeatSentMs = 0;
	private inFlightHeartbeat: Promise<boolean> | null = null;
	private hiddenSinceMs: number | null = null;
	private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
	private capacityTimer: ReturnType<typeof setInterval> | null = null;
	private idleTimer: ReturnType<typeof setInterval> | null = null;
	private cleanupListeners: (() => void) | null = null;

	idleMinutesLabel = $derived<number>(
		Math.max(
			1,
			Math.round(
				(this.idleReason === 'background'
					? this.hiddenIdleTimeoutSeconds
					: this.idleTimeoutSeconds) / 60
			)
		)
	);

	recordActivity = () => {
		const now = Date.now();
		if (now - this.lastActivityMs < 1000) return;
		this.lastActivityMs = now;
	};

	start() {
		untrack(() => {
			if (this.started || typeof window === 'undefined') return;
			this.started = true;
			this.lastActivityMs = Date.now();
			this.hiddenSinceMs = document.visibilityState === 'hidden' ? Date.now() : null;

			registerDemoClientHooks({
				onCapacityReached: (online, max) => {
					if (online > 0) this.onlineUsers = online;
					if (max > 0) this.maxUsers = max;
					this.enterCapacityWait();
				},
				onActivity: () => {
					this.lastActivityMs = Date.now();
				}
			});

			const onUserEvent = () => {
				if (!this.isIdlePaused && !this.atCapacity) {
					this.recordActivity();
				}
			};

			const onVisibilityChange = () => {
				if (document.visibilityState === 'hidden') {
					this.hiddenSinceMs = Date.now();
				} else {
					this.hiddenSinceMs = null;
					this.lastActivityMs = Date.now();
					if (!this.isIdlePaused) {
						void this.sendHeartbeat(false);
					}
				}
			};

			const onPageHide = () => {
				this.sendLeaveBeacon();
			};

			const activityEvents = ['pointerdown', 'keydown', 'wheel', 'touchstart'] as const;
			for (const ev of activityEvents) {
				window.addEventListener(ev, onUserEvent, { passive: true });
			}
			document.addEventListener('visibilitychange', onVisibilityChange);
			window.addEventListener('pagehide', onPageHide);

			this.cleanupListeners = () => {
				for (const ev of activityEvents) {
					window.removeEventListener(ev, onUserEvent);
				}
				document.removeEventListener('visibilitychange', onVisibilityChange);
				window.removeEventListener('pagehide', onPageHide);
			};

			this.startHeartbeatLoop();
			this.idleTimer = setInterval(() => this.checkIdleState(), IDLE_CHECK_INTERVAL_MS);
			void this.sendHeartbeat(true);
		});
	}

	stop() {
		untrack(() => {
			if (!this.started) return;
			this.started = false;
			this.clearTimers();
			this.cleanupListeners?.();
			this.cleanupListeners = null;
			this.isIdlePaused = false;
			this.atCapacity = false;
		});
	}

	private clearTimers() {
		if (this.heartbeatTimer) {
			clearInterval(this.heartbeatTimer);
			this.heartbeatTimer = null;
		}
		if (this.capacityTimer) {
			clearInterval(this.capacityTimer);
			this.capacityTimer = null;
		}
		if (this.idleTimer) {
			clearInterval(this.idleTimer);
			this.idleTimer = null;
		}
	}

	private startHeartbeatLoop() {
		if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
		this.heartbeatTimer = setInterval(() => {
			if (
				typeof document !== 'undefined' &&
				document.visibilityState === 'visible' &&
				!this.isIdlePaused &&
				!this.atCapacity
			) {
				void this.sendHeartbeat(false);
			}
		}, HEARTBEAT_INTERVAL_MS);
	}

	private enterCapacityWait() {
		this.atCapacity = true;
		if (this.heartbeatTimer) {
			clearInterval(this.heartbeatTimer);
			this.heartbeatTimer = null;
		}
		if (!this.capacityTimer) {
			this.capacityTimer = setInterval(() => {
				if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
					void this.sendHeartbeat(false);
				}
			}, CAPACITY_RETRY_INTERVAL_MS);
		}
	}

	private applySnapshot(data: DemoStatusPayload) {
		this.onlineUsers = Math.max(0, Number(data.online_users ?? 1));
		this.maxUsers = Math.max(1, Number(data.max_users ?? 6));
		this.lagWarning = Boolean(data.lag_warning);
		this.lagWarningThreshold = Math.max(1, Number(data.lag_warning_threshold ?? 3));
		this.activeComputeJobs = Math.max(0, Number(data.active_compute_jobs ?? 0));
		this.maxComputeJobs = Math.max(1, Number(data.max_compute_jobs ?? 2));
		if (data.idle_timeout_seconds) {
			this.idleTimeoutSeconds = Math.max(30, Number(data.idle_timeout_seconds));
		}
		if (data.hidden_idle_timeout_seconds) {
			this.hiddenIdleTimeoutSeconds = Math.max(20, Number(data.hidden_idle_timeout_seconds));
		}
	}

	async sendHeartbeat(force = false): Promise<boolean> {
		if (typeof window === 'undefined') return true;
		if (this.inFlightHeartbeat) {
			return this.inFlightHeartbeat;
		}
		const now = Date.now();
		const minGap = force ? 1_000 : MIN_HEARTBEAT_GAP_MS;
		if (now - this.lastHeartbeatSentMs < minGap) {
			return !this.atCapacity;
		}
		this.lastHeartbeatSentMs = now;

		const run = async (): Promise<boolean> => {
			try {
				const sid = getDemoSessionId();
				const res = await fetch(`${API_BASE}/quick_pp/demo/heartbeat`, {
					method: 'POST',
					credentials: 'include',
					headers: authHeaders({ 'Content-Type': 'application/json' }),
					body: JSON.stringify({ session_id: sid })
				});
				if (res.status === 429) {
					// Back off for a full heartbeat cycle if reverse-proxy rate limit was hit
					this.lastHeartbeatSentMs = Date.now() + HEARTBEAT_INTERVAL_MS;
					return false;
				}
				if (!res.ok) return false;
				const data = (await res.json()) as DemoStatusPayload;
				this.applySnapshot(data);

				if (data.admitted === false || data.at_capacity) {
					this.enterCapacityWait();
					return false;
				}

				if (this.atCapacity) {
					this.atCapacity = false;
					if (this.capacityTimer) {
						clearInterval(this.capacityTimer);
						this.capacityTimer = null;
					}
					this.lastActivityMs = Date.now();
					this.startHeartbeatLoop();
				}
				return true;
			} catch {
				return false;
			} finally {
				this.inFlightHeartbeat = null;
			}
		};

		this.inFlightHeartbeat = run();
		return this.inFlightHeartbeat;
	}

	private checkIdleState() {
		if (this.isIdlePaused || this.atCapacity || typeof document === 'undefined') return;
		// Never interrupt an in-flight calculation or report generation
		if (appStatus.isBusy) {
			this.lastActivityMs = Date.now();
			return;
		}

		const now = Date.now();
		if (
			document.visibilityState === 'hidden' &&
			this.hiddenSinceMs !== null &&
			now - this.hiddenSinceMs >= this.hiddenIdleTimeoutSeconds * 1000
		) {
			void this.pauseSession('background');
			return;
		}

		if (now - this.lastActivityMs >= this.idleTimeoutSeconds * 1000) {
			void this.pauseSession('inactivity');
		}
	}

	async pauseSession(reason: 'inactivity' | 'background') {
		if (this.isIdlePaused) return;
		this.isIdlePaused = true;
		this.idleReason = reason;
		if (this.heartbeatTimer) {
			clearInterval(this.heartbeatTimer);
			this.heartbeatTimer = null;
		}
		try {
			const sid = getDemoSessionId();
			const res = await fetch(`${API_BASE}/quick_pp/demo/leave`, {
				method: 'POST',
				credentials: 'include',
				headers: authHeaders({ 'Content-Type': 'application/json' }),
				body: JSON.stringify({ session_id: sid })
			});
			if (res.ok) {
				const data = (await res.json()) as DemoStatusPayload;
				this.applySnapshot(data);
			}
		} catch {
			// Ignore network errors while releasing idle slot
		}
	}

	async resumeSession() {
		this.isReconnecting = true;
		try {
			this.lastActivityMs = Date.now();
			this.hiddenSinceMs = null;
			const admitted = await this.sendHeartbeat(true);
			this.isIdlePaused = false;
			if (admitted) {
				this.startHeartbeatLoop();
			}
		} finally {
			this.isReconnecting = false;
		}
	}

	async retryCapacityCheck() {
		this.isReconnecting = true;
		try {
			await this.sendHeartbeat(true);
		} finally {
			this.isReconnecting = false;
		}
	}

	private sendLeaveBeacon() {
		if (typeof navigator === 'undefined' || !navigator.sendBeacon) return;
		try {
			const sid = getDemoSessionId();
			if (!sid) return;
			const blob = new Blob([JSON.stringify({ session_id: sid })], {
				type: 'application/json'
			});
			navigator.sendBeacon(`${API_BASE}/quick_pp/demo/leave`, blob);
		} catch {
			// Best-effort beacon on tab close
		}
	}
}

export const demoSession = new DemoSessionManager();
