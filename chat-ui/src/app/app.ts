import { Component, ElementRef, ViewChild, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

interface ModelResult {
  model: string;
  predicted_price: number | null;
  rmse: number | null;
  mape: number | null;
  dir_accuracy: number | null;
  error?: string;
}

interface ChatResponse {
  synthesis: string;
  model_results: ModelResult[];
  current_price: number;
  last_date: string;
  table: string;
}

interface Message {
  role: 'user' | 'assistant';
  text: string;
  response?: ChatResponse;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  @ViewChild('messagesEnd') messagesEnd!: ElementRef;

  messages = signal<Message[]>([]);
  input = '';
  loading = signal(false);
  error = signal('');

  private readonly API = '/api';

  constructor(private http: HttpClient) {}

  send() {
    const query = this.input.trim();
    if (!query || this.loading()) return;

    this.messages.update(msgs => [...msgs, { role: 'user', text: query }]);
    this.input = '';
    this.loading.set(true);
    this.error.set('');

    this.http.post<ChatResponse>(`${this.API}/chat`, { query }).subscribe({
      next: (res) => {
        this.messages.update(msgs => [...msgs, {
          role: 'assistant',
          text: res.synthesis,
          response: res
        }]);
        this.loading.set(false);
        setTimeout(() => this.scrollToBottom(), 50);
      },
      error: (err) => {
        this.error.set(err?.error?.detail ?? 'Backend error — is the server running?');
        this.loading.set(false);
      }
    });

    setTimeout(() => this.scrollToBottom(), 50);
  }

  onKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.send();
    }
  }

  private scrollToBottom() {
    this.messagesEnd?.nativeElement?.scrollIntoView({ behavior: 'smooth' });
  }
}
