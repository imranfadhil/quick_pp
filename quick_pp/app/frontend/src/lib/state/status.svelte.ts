import { untrack } from 'svelte';

export type StatusLevel = 'idle' | 'busy' | 'success' | 'warning' | 'error';

export interface ActiveTask {
	id: string;
	label: string;
	detail?: string;
	progress?: number | null;
	startedAt: number;
}

export interface StatusLogItem {
	id: string;
	timestamp: string;
	level: 'info' | 'success' | 'warning' | 'error';
	message: string;
	detail?: string;
	durationMs?: number;
}

const MAX_HISTORY = 40;

function nowTimeStr(): string {
	return new Date().toLocaleTimeString([], {
		hour: '2-digit',
		minute: '2-digit',
		second: '2-digit',
		hour12: false
	});
}

class AppStatusState {
	activeTasks = $state<ActiveTask[]>([]);
	level = $state<StatusLevel>('idle');
	message = $state<string>('Ready');
	detail = $state<string | null>(null);
	lastDurationMs = $state<number | null>(null);
	history = $state<StatusLogItem[]>([]);
	logPopoverOpen = $state<boolean>(false);

	private taskCounter = 0;

	isBusy = $derived<boolean>(this.activeTasks.length > 0);
	currentTask = $derived<ActiveTask | null>(
		this.activeTasks.length > 0 ? this.activeTasks[this.activeTasks.length - 1] : null
	);
	displayLevel = $derived<StatusLevel>(this.activeTasks.length > 0 ? 'busy' : this.level);
	displayMessage = $derived<string>(this.currentTask ? this.currentTask.label : this.message);
	displayDetail = $derived<string | null>(
		this.currentTask ? (this.currentTask.detail ?? null) : this.detail
	);
	displayProgress = $derived<number | null>(this.currentTask?.progress ?? null);

	startTask(label: string, detail?: string, progress: number | null = null): string {
		const id = `task-${++this.taskCounter}-${Date.now()}`;
		const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
		untrack(() => {
			this.activeTasks = [
				...this.activeTasks,
				{
					id,
					label,
					detail,
					progress,
					startedAt: now
				}
			];
		});
		return id;
	}

	updateTask(
		id: string,
		updates: { label?: string; detail?: string; progress?: number | null }
	): void {
		untrack(() => {
			this.activeTasks = this.activeTasks.map((t) =>
				t.id === id
					? {
							...t,
							label: updates.label !== undefined ? updates.label : t.label,
							detail: updates.detail !== undefined ? updates.detail : t.detail,
							progress: updates.progress !== undefined ? updates.progress : t.progress
						}
					: t
			);
		});
	}

	endTask(
		id: string,
		outcome?: {
			level?: StatusLevel;
			message?: string;
			detail?: string | null;
			silent?: boolean;
		}
	): number {
		return untrack(() => {
			const task = this.activeTasks.find((t) => t.id === id);
			const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
			const durationMs = task ? Math.max(1, Math.round(now - task.startedAt)) : 0;

			this.activeTasks = this.activeTasks.filter((t) => t.id !== id);

			if (outcome?.silent) {
				return durationMs;
			}

			const nextLevel = outcome?.level ?? 'success';
			const nextMsg =
				outcome?.message ??
				(task ? task.label.replace(/\.{2,}$/, '') : this.message);
			const nextDetail = outcome?.detail !== undefined ? outcome.detail : (task?.detail ?? null);

			this.level = nextLevel;
			this.message = nextMsg;
			this.detail = nextDetail;
			this.lastDurationMs = durationMs;

			if (nextLevel !== 'idle') {
				this.pushHistory({
					level:
						nextLevel === 'error'
							? 'error'
							: nextLevel === 'warning'
								? 'warning'
								: nextLevel === 'success'
									? 'success'
									: 'info',
					message: nextMsg,
					detail: nextDetail ?? undefined,
					durationMs
				});
			}

			return durationMs;
		});
	}

	setStatus(
		level: StatusLevel,
		message: string,
		detail: string | null = null,
		durationMs: number | null = null
	) {
		this.level = level;
		this.message = message;
		this.detail = detail;
		this.lastDurationMs = durationMs;
		if (level !== 'idle' && level !== 'busy') {
			this.pushHistory({
				level:
					level === 'error'
						? 'error'
						: level === 'warning'
							? 'warning'
							: level === 'success'
								? 'success'
								: 'info',
				message,
				detail: detail ?? undefined,
				durationMs: durationMs ?? undefined
			});
		}
	}

	info(message: string, detail?: string) {
		this.setStatus('idle', message, detail ?? null, null);
		this.pushHistory({
			level: 'info',
			message,
			detail
		});
	}

	success(message: string, detail?: string, durationMs?: number) {
		this.setStatus('success', message, detail ?? null, durationMs ?? this.lastDurationMs);
	}

	warn(message: string, detail?: string) {
		this.setStatus('warning', message, detail ?? null, null);
	}

	error(message: string, detail?: string) {
		this.setStatus('error', message, detail ?? null, null);
	}

	clearHistory() {
		this.history = [];
	}

	private pushHistory(item: Omit<StatusLogItem, 'id' | 'timestamp'>) {
		const entry: StatusLogItem = {
			id: `log-${++this.taskCounter}-${Date.now()}`,
			timestamp: nowTimeStr(),
			...item
		};
		this.history = [entry, ...this.history].slice(0, MAX_HISTORY);
	}
}

export const appStatus = new AppStatusState();

