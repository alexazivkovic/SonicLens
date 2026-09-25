import type { ModelCard } from '../types'

export function About({ model, onClose }: { model: ModelCard | null; onClose: () => void }) {
  return (
    <section className="card about" aria-labelledby="about-title">
      <div className="about-head">
        <h2 id="about-title">About SonicLens</h2>
        <button type="button" className="button" onClick={onClose}>
          Close
        </button>
      </div>
      <p>
        SonicLens is a free, open-source tool for researchers, developers and music enthusiasts. Upload an audio file
        and get two kinds of descriptors:
      </p>
      <ul>
        <li>
          <strong>Low-level descriptors</strong> (tempo, key, loudness, spectral, timbral, rhythmic and tonal
          statistics), computed with established DSP algorithms from the{' '}
          <a href="https://essentia.upf.edu/" target="_blank" rel="noreferrer">Essentia</a> library.
        </li>
        <li>
          <strong>Perceptual descriptors</strong> (energy, danceability, valence, acousticness, instrumentalness,
          liveness, speechiness), predicted by a machine-learning model from the low-level descriptors of the central
          30 seconds of the track.
        </li>
      </ul>
      <h3>Training data</h3>
      <p>
        The model was trained on audio clips from the{' '}
        <a href="https://github.com/mdeff/fma" target="_blank" rel="noreferrer">Free Music Archive (FMA)</a> dataset
        {model && <> ({model.training_data.n_tracks_used.toLocaleString('en')} tracks)</>}. The targets are the Echo
        Nest audio descriptors distributed with FMA. FMA metadata is licensed under{' '}
        <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>.
        {model && <> Citation: {model.training_data.citation}</>}
      </p>
      <p>
        Accuracy was measured on a held-out test set of
        {model ? ` ${model.training_data.n_test.toLocaleString('en')} tracks by ${model.training_data.n_artists_test} ` : ' '}
        artists the model never saw; the test R² of each descriptor is shown next to its value.
      </p>
      <h3>Limitations</h3>
      {model ? (
        <ul>
          {model.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      ) : (
        <p className="muted">The model card is not available right now.</p>
      )}
      <h3>Privacy and license</h3>
      <p>
        Uploaded audio is analysed and deleted immediately; only the descriptors and a SHA-256 hash of the file are
        stored, so the same file is answered from the cache next time. SonicLens is licensed under the{' '}
        <a href="https://www.gnu.org/licenses/agpl-3.0.html" target="_blank" rel="noreferrer">GNU AGPL v3.0</a>.
      </p>
    </section>
  )
}
