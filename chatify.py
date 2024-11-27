# INITS + IMPORTS
import gradio as gr
from typing import List
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY=api_key="yen.k@northeastern.edu:79wlaJxIMpzvkt61S8Xi"
assert BASE_URL is not None
assert API_KEY is not None

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

load_dotenv() 

# Set up the client id and client secret (you can find the client secret 
# from the spotify dev page)
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = 'playlist-modify-public user-modify-playback-state user-top-read'

# Create a spotify object passing the client id, client secret, 
# redirct url (which doesn't matter, just set it as your local host
# as shown below), and scope
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE,
                                               cache_path=".cache-<your-username>"))

# TODO define the data classes

GENRES = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'work-out', 'world-music']

class SpotifyAgent:
    conversation: List[dict]
    client: OpenAI
    spotify: SpotifyOAuth
    playlist_id: str
    seed_artists: List[str]
    seed_tracks: List[str]
    seed_genres: List[str]
    possible_genres: List[str]
    # MOOD DETERMINATORS
    target_valence: float # 0.0 to 1.0 = musical positiveness conveyed by a track. high valence = positive
    target_energy: float # 0.0 to 1.0 = perceptual measure of intensity and activity. high energy = fast, loud, noisy
    target_danceability: float # 0.0 to 1.0 = how suitable a track is for dancing. high danceability = easy to dance to

    system_prompt = """
You are a helpful Spotify chatbot. Respond to queries with a single Python block of code that uses
the following function:
def add_to_seed_artists(artist_name):
    ...
def add_to_seed_tracks(track_name):
    ...
def add_to_seed_genres(genre_name):
    ...
def add_tracks_to_playlist(track_titles_or_uris):
    ...
# valence, energy, danceability are floats between 0 and 1
def modify_mood(valence: float, energy: float, danceability: float):
    ...
Return the result in a variable called result.
Do not redefine the functin stubs just use this existing method.
Don't correct their questions.
"""
    few_shot_prompt = "I like the artist bobbin."
    few_shot_response = """
```python
result = add_to_seed_artists("bobbin")
print(result)
```
"""
    few_shot_prompt2 = "Get me reccomendations."
    few_shot_response2 = """
```python
result = get_recommendations()
print(result)
```
"""
    few_shot_response_another = """
Here are some recommendations for you:
Love Story uri: spotify:track:1CkvWZme3pRgbzaxZnTl5X
Blank Space uri: spotify:track:1p80LdxRV74UKvL8gnD7ky
"""
    few_shot_prompt3 = "Add the second song to the playlist."
    few_shot_response3 = """
```python
result = add_tracks_to_playlist("spotify:track:1p80LdxRV74UKvL8gnD7ky")
print(result)
```
"""
    few_shot_prompt4 = "Add the song True Blue to the playlist."
    few_shot_response4 = """
```python
result = add_tracks_to_playlist("True Blue")
print(result)
```
"""
    # clear playlist if exists, else create a new one
    def create_playlist(self):
        user_id = self.spotify.current_user()['id']  
        playlist_name = 'Spotify Chatbot Session'
        playlist_description = 'A playlist created by the Spotify Chatbot.'
        # Get the current user's playlists
        playlists = self.spotify.current_user_playlists()

        # Look for the playlist by name
        existing_playlist = None
        for playlist in playlists['items']:
            if playlist['name'] == playlist_name:
                existing_playlist = playlist
                break

        if existing_playlist:
            # If the playlist exists, take action: delete all tracks from it
            playlist_id = existing_playlist['id']
            print(f"Found existing playlist '{playlist_name}', clearing tracks...")
            
            # Get all tracks
            track_uris = []
            results = self.spotify.playlist_items(playlist_id)
            for item in results['items']:
                if item and item['track']:
                    track_uris.append(item['track']['uri'])
            # Remove all tracks
            if track_uris:
                self.spotify.playlist_remove_all_occurrences_of_items(playlist_id, track_uris)
                print(f"All tracks removed from playlist '{playlist_name}'.")
            return playlist_id
        else:
            # If the playlist does not exist, create a new one
            print(f"Playlist '{playlist_name}' does not exist. Creating a new one...")
            res = self.spotify.user_playlist_create(user=user_id, 
                                    name=playlist_name, 
                                    public=True, 
                                    description=playlist_description)
            print(f"Playlist '{playlist_name}' created successfully.")
            return res['id']

    def get_track_uri(self, track_name):
        results = self.spotify.search(q='track:' + track_name, type='track')
        items = results['tracks']['items']
        if items:
            return items[0]['uri']
        return None
    # Add one song or multiple songs to current playlist
    def add_tracks_to_playlist(self, track_titles_or_uris):
        if type(track_titles_or_uris) is not list:
            track_titles_or_uris = [track_titles_or_uris]

        track_uris = []
        for track in track_titles_or_uris:
            if track.startswith('spotify:track:'):
                track_uris.append(track)
            else: 
                result = self.get_track_uri(track)
                if result:
                    track_uris.append(result)
        spotify.playlist_add_items(self.playlist_id, track_uris)


    # Get recommendations based on seeds
    def get_recommendations(self):
        limit = 10
        # Regardless of mood values you need seeds to get recommendations
        if len(self.seed_artists) == 0 and len(self.seed_tracks) == 0 and len(self.seed_genres) == 0:
            if self.target_valence == -1 and self.target_energy == -1 and self.target_danceability == -1:
                print("No seeds or Mood provided. Returning top tracks.")
                return spotify.current_user_top_tracks(limit=limit, offset=0, time_range='short_term')['items']

            # Add top artist to seeds
            print("Mood Provided, but no seeds. Using favorite song as seed.")
            top_track_uri = spotify.current_user_top_tracks(limit=1, offset=0, time_range='short_term')['items'][0]['uri']
            self.seed_tracks.append(top_track_uri)
            recommendations = spotify.recommendations(
            seed_artists=self.seed_artists,
            seed_tracks=[top_track_uri],
            seed_genres=self.seed_genres,
            target_valence=self.target_valence,
            target_energy=self.target_energy,
            target_danceability=self.target_danceability,
            limit=limit
        )
        
        recommendations = spotify.recommendations(
            seed_artists=self.seed_artists,
            seed_tracks=self.seed_tracks,
            seed_genres=self.seed_genres,
            target_valence=self.target_valence,
            target_energy=self.target_energy,
            target_danceability=self.target_danceability,
            limit=limit
        )

        return recommendations['tracks']
    
    def add_to_seed_artists(self, artist_name):
        # Perform a search query for the artist
        results = spotify.search(q='artist:' + artist_name, type='artist')
        
        items = results['artists']['items']
        if items:
            artist = items[0]
            self.seed_artists.append(artist['uri']) # add first artist
        
         # TODO Figure out error handling

    def add_to_seed_tracks(self, track_name):
        results = spotify.search(q='track:' + track_name, type='track')
        items = results['tracks']['items']
        if items:
            track = items[0]
            self.seed_tracks.append(track['uri'])
    
    def add_to_seed_genres(self, genre_name):
        if genre_name.lower() in self.possible_genres:
            self.seed_genres.append(genre_name.lower())

    def modify_mood(self, valence: float, energy: float, danceability: float):
        self.target_valence = valence
        self.target_energy = energy
        self.target_danceability = danceability

    def get_artist(self, artist_uri): 
        return spotify.artist(artist_uri)
    
    def get_track(self, track_uri):
        return spotify.track(track_uri)

    def extract_code(self, resp_text):
        code_start = resp_text.find("```")
        code_end = resp_text.rfind("```")
        if code_start == -1 or code_end == -1:
            return "pass"
        
        return resp_text[code_start + 3 + 7:code_end]
    
    def run_code(self, code_text):
        globals = { 
            "add_tracks_to_playlist": self.add_tracks_to_playlist, 
            "get_recommendations": self.get_recommendations,
            "add_to_seed_artists": self.add_to_seed_artists,
            "add_to_seed_tracks": self.add_to_seed_tracks,
            "add_to_seed_genres": self.add_to_seed_genres,
            "modify_mood": self.modify_mood,
        }
        exec(code_text, globals)
        return globals["result"]
    
    # TODO add the types??
    def say(self, user_message: str):
        if len(self.conversation) > 20:
            self.conversation.pop(1)
            self.conversation.pop(2)
        # Add the user message to the conversation.
        self.conversation.append({"role": "user", "content": user_message})
        
        # Get the response from the model.
        resp = self.client.chat.completions.create(
            messages = self.conversation,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0)
        
        print('responsed')
        resp_text = resp.choices[0].message.content

        print('RESP TEXT', resp_text)
        self.conversation.append({"role": "system", "content": resp_text })
        code_text = self.extract_code(resp_text)
        # TODO if prompt was relating to add seed artists, delete from conversation
        # store conversation to display
        res = self.run_code(code_text)
        
        message = ''
        if "get_recommendations" in code_text:
            system_recc_resp = "Here are some recommendations for you:"
            user_recc_resp = "Here are some recommendations for you:"
            for track in res:
                system_recc_resp += '\n' + track['name'] + ' uri: ' + track['uri']
                user_recc_resp += '\n' + track['name'] + ' by ' + track['artists'][0]['name']
            self.conversation.append({"role": "system", "content": system_recc_resp})
            message += user_recc_resp
        if "add_to_seed_artists" in code_text:
            new_artist = self.seed_artists[-1]
            message += f"Added {self.get_artist(new_artist)['name']} to seed artists."
        if "add_to_seed_tracks" in code_text:
            new_track = self.seed_tracks[-1]
            message += f"Added {self.get_track(new_track)['name']} to seed tracks."
        if "add_to_seed_genres" in code_text:
            new_genre = self.seed_genres[-1]
            message += f"Added {new_genre} to seed genres."
        if "modify_mood" in code_text:
            message += f"Modified mood to valence: {self.target_valence}, energy: {self.target_energy}, danceability: {self.target_danceability}."
        if "add_tracks_to_playlist" in code_text:
            message += "Added tracks to playlist." # some how also add a spotify iframe... 
        return message
    
    def __init__(self, client: OpenAI, spotify: SpotifyOAuth):
        self.client = client
        self.spotify = spotify
        self.playlist_id = self.create_playlist() # Create a playlist for the session
        # self.possible_genres = spotify.recommendation_genre_seeds()['genres'] request too slow
        self.possible_genres = GENRES
        self.conversation = [{ "role": "system", "content": self.system_prompt },
                             { "role": "user", "content": self.few_shot_prompt },
                             { "role": "system", "content": self.few_shot_response}, 
                             { "role": "user", "content": self.few_shot_prompt2 },
                             { "role": "system", "content": self.few_shot_response2 },
                             { "role": "system", "content": self.few_shot_response_another},
                             { "role": "user", "content": self.few_shot_prompt3 },
                             { "role": "system", "content": self.few_shot_response3}]
        # TODO? add back fewshot response 4?
        self.seed_artists = []
        self.seed_tracks = []
        self.seed_genres = []
        self.target_valence = -1
        self.target_energy = -1
        self.target_danceability = -1
        # self.current_mood = 0

spotify_agent = SpotifyAgent(client=client, spotify=spotify)


def spotify_chat(user_message, history):
    bot_response = spotify_agent.say(user_message)
    #bot_response = spotify_agent.conversation[-1]['content']  # Last bot message
    history.append((user_message, bot_response))
    return history

# Gradio Interface
with gr.Blocks() as spotify_chatbot:
    gr.Markdown("## 🎵 Spotify Chatbot")
    gr.Markdown("Chat with your Spotify assistant to get recommendations, create playlists, and more!")

    # Chatbot with conversation history
    chatbot = gr.Chatbot(label="Spotify Assistant")

    # User input and send button
    user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    send_button = gr.Button("Send")

    # Button interaction logic
    def interact(message, chat_history):
        return spotify_chat(message, chat_history), ""

    send_button.click(interact, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

# Launch the Gradio app
spotify_chatbot.launch()