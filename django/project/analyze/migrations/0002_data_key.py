# Generated by Django 3.0.2 on 2020-02-12 14:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyze', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='key',
            field=models.FloatField(default=0),
        ),
    ]
